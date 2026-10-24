"""Mean of the cross-fitted training residual x - x_r (what the residual diffusion learns to centre on) on ~200
training cubes from the cache, plus the same on the 200 gate cubes from the val cache (full regressor)."""
import json, os, sys
import numpy as np
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.xr_cache import cache_path

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
CACHE = R + '/uq/d8/xr_cache'
meta = json.load(open(CACHE + '/meta.json'))
split = json.load(open(meta['split_path']))
h1 = set(split['half1']); h2 = set(split['half2'])
sig = meta['sigma_r']
print('DATA_ROOT', DATA_ROOT, 'PRIOR', PRIOR, 'prior == meta prior', os.path.abspath(PRIOR) == meta['prior_path'])


def stats(ds, split_name, idxs, label):
    acc_sum = np.zeros(64); acc_sq = np.zeros(64); n_pix = 0
    per_cube_mean = []; per_cube_band = []; halves = []; abs_lo = 0; tot = 0
    ss = 0.0; nn = 0
    for i in idxs:
        f = ds.img_list[i]
        rel = os.path.relpath(f, DATA_ROOT)
        x, _ = ds.raw_item(i)
        xr = np.load(cache_path(CACHE, split_name, rel)).astype(np.float64)
        r = x.astype(np.float64) - xr                           # [H, W, 64] on [-1, 1]
        acc_sum += r.sum((0, 1)); acc_sq += (r ** 2).sum((0, 1)); n_pix += r.shape[0] * r.shape[1]
        per_cube_mean.append(r.mean()); per_cube_band.append(r.mean((0, 1)))
        halves.append(1 if rel in h1 else (2 if rel in h2 else 0))
        ss += float((r ** 2).sum()); nn += r.size
    pcm = np.array(per_cube_mean); pcb = np.array(per_cube_band); halves = np.array(halves)
    band_mean = acc_sum / n_pix
    rms = np.sqrt(ss / nn)
    n = len(idxs)
    print(f'\n== {label}: {n} cubes')
    print(f'  RMS(x - x_r) {rms:.6f}  (meta sigma_r {meta["sigma_r"]:.6f}, sigma_full {meta["sigma_full"]:.6f})')
    print(f'  global mean {pcm.mean():+.6f} [-1,1] = {pcm.mean()/sig:+.4f} sigma_r = {pcm.mean()/2*100:+.4f} pct;'
          f'  SE over cubes {pcm.std(ddof=1)/np.sqrt(n):.6f} -> t = {pcm.mean()/(pcm.std(ddof=1)/np.sqrt(n)):+.2f}')
    bse = pcb.std(0, ddof=1) / np.sqrt(n)
    t = band_mean / bse
    print(f'  per-band mean (pct): min {band_mean.min()/2*100:+.4f} max {band_mean.max()/2*100:+.4f}; |t|>3: {int((np.abs(t) > 3).sum())} of 64')
    print('  8-band blocks (pct):', [round(float(band_mean[j:j+8].mean() / 2 * 100), 4) for j in range(0, 64, 8)])
    print('  8-band blocks t:', [round(float(t[j:j+8].mean()), 2) for j in range(0, 64, 8)])
    print(f'  squared mean share of MS: {float((band_mean**2).mean() / (ss/nn)):.5f}  (per-cube per-band offsets share {float((pcb**2).mean()/(ss/nn)):.4f})')
    if (halves > 0).any():
        for h in (1, 2):
            s = halves == h
            print(f'  half {h} cubes ({int(s.sum())}, centred by R{3-h}): mean {pcm[s].mean()/2*100:+.4f} pct (SE {pcm[s].std(ddof=1)/np.sqrt(s.sum())/2*100:.4f})')
    return band_mean, pcm, pcb


tr = HFDResidualData(data_path=DATA_ROOT, split='train', sensor_down_sample_rate=8, prior_path=PRIOR)
print('train n', len(tr))
idx = np.linspace(0, len(tr) - 1, 200).round().astype(int).tolist()
bm_tr, pcm_tr, pcb_tr = stats(tr, 'train', idx, 'train (cross-fitted, 200 evenly spaced)')

te = HFDResidualData(data_path=DATA_ROOT, split='test', sensor_down_sample_rate=8, prior_path=PRIOR)
gate = set(json.load(open(REPO + '/tests/data/val_subset_200.json'))['files'])
gidx = [i for i, f in enumerate(te.img_list) if os.path.relpath(f, DATA_ROOT) in gate]
print('gate idx found', len(gidx))
bm_g, pcm_g, pcb_g = stats(te, 'val', gidx, 'gate (val cache, full regressor)')
midx = [i for i, f in enumerate(te.img_list) if os.path.relpath(f, DATA_ROOT) not in gate]
bm_m, _, _ = stats(te, 'val', midx[::4], 'monitoring (val cache, every 4th)')
print('\ncorr(band means train, gate):', float(np.corrcoef(bm_tr, bm_g)[0, 1]))
np.savez('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/provenance/p5_bandmeans.npz', train=bm_tr, gate=bm_g, mon=bm_m)
