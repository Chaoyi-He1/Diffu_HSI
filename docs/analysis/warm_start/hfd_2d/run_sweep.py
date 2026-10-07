"""Step 2 (GPU 0, <2.5 GB): warm-start / truncated sampling sweep on the HFD 64-band checkpoint.

For every test item (cube x, sensor y) and noise seed:
  starting estimates x0_hat(y) -- computed from y only, never from x:
    lin   : A (mu + K (y - R^T mu)), clipped to [-1,1]           (Gaussian-prior MMSE, 31 -> 64 bands)
    gauss : A (mu_post + L z1)  with L L^T = C - K R^T C         (posterior sample of the surrogate)
    self  : 5-step deterministic DDIM (t = 999,799,599,400,200 -> 0) from N(0,I), clipped  (5 NFE)
    mean  : A mu                                                 (prior mean, ignores y)
  warm start: x_t0 = sqrt(ab_t0) x0_hat + sqrt(1-ab_t0) z, then
    linreg: as lin but with s = 0.1 (a worse, noise-robust linear estimate; extra init)
    DDPM ancestral steps t0, t0-1, ..., 0   (t0 in 600,400,300,200,100,50 + extra 20,10; NFE = t0+1)
    DDIM eta=0 with S evenly spaced steps t0 -> 0 (t0 in 400,200; S in 10,20; NFE = S)
  baselines: full DDPM from x_999 ~ N(0,I) (1000 NFE), full DDIM 50 / 100 steps, each estimate alone.
Common random numbers: the re-noising draw z and the DDPM step noises depend on
(seed, batch, sampler, t0, steps) but not on the init, so inits are compared on the same noise.
Records are appended to records_<set>.jsonl; re-running resumes (skips finished keys).
"""
import os, sys, json, time, zlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--set', default='primary', choices=['primary', 'secondary'])
ap.add_argument('--seeds', default='0,1')
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--cpu', action='store_true')
ap.add_argument('--reduced', action='store_true', help='reduced sweep (used for the secondary set): inits lin/mean/linreg, DDPM t0 10/50/200/400, DDIM10 t0 200/400, full DDIM50 + full DDPM')
ap.add_argument('--clip-x0', dest='clip_x0', action='store_true', help='extra runs with the x0 prediction clipped to [-1,1] inside the sampler (config suffix -clip); the repo sampler does not clip')
ap.add_argument('--smoke', action='store_true', help='tiny run (1 batch of 2, few configs) to records_smoke.jsonl')
args = ap.parse_args()
seeds = [int(s) for s in args.seeds.split(',')]

torch.set_num_threads(8)
dev = torch.device('cpu') if args.cpu else pick_device()
print('device', dev, flush=True)
model, betas, meta = load_model(dev, sdpa=True)
print('checkpoint', meta, flush=True)
sch = Schedule(betas, dev)
ab = sch.ab.tolist(); alphas = sch.alphas.tolist(); bet = sch.betas.tolist(); pv = sch.post_var.tolist(); abp = sch.ab_prev.tolist()

# ---------------- data ----------------
ts_ = json.load(open(os.path.join(OUT, 'testsets.json')))
files = ts_[args.set]
tr, ev = make_datasets()
idx = {p: (ev, i) for i, p in enumerate(ev.img_list)}
idx.update({p: (tr, i) for i, p in enumerate(tr.img_list)})
X, Y = [], []
for p in files:
    ds, i = idx[p]
    g, s = ds[i]                       # the snapshot's own __getitem__: [H,W,64] cube, [H,W,30] sensor
    X.append(torch.tensor(g, dtype=torch.float32).permute(2, 0, 1))
    Y.append(torch.tensor(s, dtype=torch.float32).permute(2, 0, 1))
X = torch.stack(X); Y = torch.stack(Y)
print('items', X.shape, Y.shape, 'x range', X.min().item(), X.max().item(), flush=True)

pr = np.load(os.path.join(OUT, 'prior.npz'))
P = {k: torch.tensor(pr[k], dtype=torch.float64, device=dev) for k in ('mu', 'R', 'A', 'K', 'L')}
print('prior s =', float(pr['s']), flush=True)
# extra init 'linreg': the same linear MMSE with a much larger regulariser (s = 0.1, held-out RMSE ~1.75 %),
# i.e. a deliberately worse but noise-robust linear estimate, to see how estimate quality moves the usable t0
S_LINREG = 0.1
_C, _R = pr['C'], pr['R']
P['K_reg'] = torch.tensor(np.linalg.solve(_R.T @ _C @ _R + S_LINREG ** 2 * np.eye(30), (_C @ _R).T).T,
                          dtype=torch.float64, device=dev)


def lin_mean31(y, K='K'):
    """y [B,30,H,W] float32 -> posterior mean [B,H,W,31] float64 (unclipped)."""
    yy = y.permute(0, 2, 3, 1).double()
    return P['mu'] + (yy - P['mu'] @ P['R']) @ P[K].T


def to64(x31):
    """[B,H,W,31] -> [B,64,H,W] float32 via the loader's interpolation matrix."""
    return (x31 @ P['A'].T).permute(0, 3, 1, 2).float()


def gen(*key):
    g = torch.Generator(device=dev)
    g.manual_seed(zlib.crc32('|'.join(str(k) for k in key).encode()))
    return g


def randn(shape, g):
    return torch.randn(shape, generator=g, device=dev, dtype=torch.float32)


def eps_model(x, y, t):
    tt = torch.full((x.shape[0],), int(t), device=dev, dtype=torch.long)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16 if dev.type == 'cuda' else torch.bfloat16,
                                         enabled=(dev.type == 'cuda')):
        return model(x, y, tt).float()


def ddpm(x, y, t0, g, clip=False):
    for t in range(t0, -1, -1):
        e = eps_model(x, y, t)
        if clip:   # posterior mean from the clipped x0 prediction (DDPM 'clip_denoised')
            x0 = ((x - math.sqrt(1 - ab[t]) * e) / math.sqrt(ab[t])).clamp(-1, 1)
            x = (bet[t] * math.sqrt(abp[t]) * x0 + (1 - abp[t]) * math.sqrt(alphas[t]) * x) / (1 - ab[t])
        else:      # the repo's update
            x = (x - bet[t] / math.sqrt(1 - ab[t]) * e) / math.sqrt(alphas[t])
        if t > 0:
            x = x + math.sqrt(pv[t]) * randn(x.shape, g)
    return x


def ddim_ts(t0, S):
    return [int(round(t0 * (1 - i / S))) for i in range(S)]   # e.g. 400,360,...,40 ; 999,979,...,20


def ddim(x, y, ts, clip=False):
    for i, t in enumerate(ts):
        e = eps_model(x, y, t)
        x0 = (x - math.sqrt(1 - ab[t]) * e) / math.sqrt(ab[t])
        if clip:   # clip x0 and re-derive eps from it
            x0 = x0.clamp(-1, 1)
            e = (x - math.sqrt(ab[t]) * x0) / math.sqrt(1 - ab[t])
        a_next = ab[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = math.sqrt(a_next) * x0 + math.sqrt(1 - a_next) * e
    return x


def sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()


# ---------------- configs ----------------
INITS = ['lin', 'gauss', 'self', 'mean', 'linreg']
DDPM_T0 = [10, 20, 50, 100, 200, 300, 400, 600]
DDIM_TR = [(400, 10), (400, 20), (200, 10), (200, 20)]
SELF_NFE = 5
FULL_DDIM = (50, 100)
RUN_FULL_DDPM = True
SWEEP_INITS = INITS
if args.reduced:
    SWEEP_INITS = ['lin', 'mean', 'linreg']
    DDPM_T0, DDIM_TR, FULL_DDIM = [10, 50, 200, 400], [(400, 10), (200, 10)], (50,)
if args.smoke:
    DDPM_T0, DDIM_TR, FULL_DDIM, RUN_FULL_DDPM = [10], [(200, 10)], (50,), False
    files = files[:2]; X = X[:2]; Y = Y[:2]

rec_path = os.path.join(OUT, 'records_%s.jsonl' % ('smoke' if args.smoke else args.set))
done = set()
if os.path.exists(rec_path):
    for line in open(rec_path):
        r = json.loads(line)
        done.add((r['config'], r['seed'], r['batch']))
print('resuming with', len(done), 'finished (config, seed, batch) keys', flush=True)
fout = open(rec_path, 'a')


def record(cfg, seed, b, items, xhat, x, nfe, sec, init, t0, sampler):
    rm, sa, ps = metrics(xhat, x)
    r = dict(config=cfg, seed=seed, batch=b, items=items, init=init, t0=t0, sampler=sampler, nfe=nfe,
             sec_per_item=sec / len(items), rmse=rm.tolist(), sam=sa.tolist(), psnr=ps.tolist())
    fout.write(json.dumps(r) + '\n'); fout.flush()
    print('  %-22s seed %d batch %d  NFE %4d  RMSE %.3f%%  SAM %.3f  PSNR %.2f  (%.2fs/item)' % (
        cfg, seed, b, nfe, rm.mean(), sa.mean(), ps.mean(), sec / len(items)), flush=True)


B = args.batch
nb = (len(files) + B - 1) // B
T_START = time.time()
for seed in seeds:
    for b in range(nb):
        sl = slice(b * B, min((b + 1) * B, len(files)))
        items = list(range(sl.start, sl.stop))
        x = X[sl].to(dev); y = Y[sl].to(dev)
        print('== seed %d batch %d items %s  (elapsed %.0fs)' % (seed, b, items, time.time() - T_START), flush=True)
        # ---- starting estimates (y only) ----
        sync(); t = time.time()
        m31 = lin_mean31(y)
        est = {'lin': to64(m31).clamp(-1, 1)}
        sync(); est_sec = {'lin': time.time() - t}
        t = time.time()
        z1 = torch.randn(m31.shape, generator=gen(seed, b, 'gauss-z1'), device=dev, dtype=torch.float64)
        est['gauss'] = to64(m31 + z1 @ P['L'].T)
        sync(); est_sec['gauss'] = time.time() - t
        est['mean'] = to64(P['mu'].expand_as(m31)).contiguous(); est_sec['mean'] = 0.0
        t = time.time()
        est['linreg'] = to64(lin_mean31(y, 'K_reg')).clamp(-1, 1)
        sync(); est_sec['linreg'] = time.time() - t
        sync(); t = time.time()
        est['self'] = ddim(randn(x.shape, gen(seed, b, 'self-xT')), y, ddim_ts(999, SELF_NFE)).clamp(-1, 1)
        sync(); est_sec['self'] = time.time() - t
        est_nfe = {'lin': 0, 'gauss': 0, 'mean': 0, 'linreg': 0, 'self': SELF_NFE}
        for k in INITS:
            cfg = 'est-%s' % k
            if (cfg, seed, b) not in done:
                record(cfg, seed, b, items, est[k], x, est_nfe[k], est_sec[k], k, None, 'none')

        def warm(init, t0, g):
            return math.sqrt(ab[t0]) * est[init] + math.sqrt(1 - ab[t0]) * randn(x.shape, g)

        if args.clip_x0:
            # same noise keys as the unclipped runs -> paired comparison clip vs no clip
            jobs = [('full-ddpm-clip', 'noise', 999, 'ddpm', None), ('full-ddim50-clip', 'noise', 999, 'ddim', 50),
                    ('full-ddim100-clip', 'noise', 999, 'ddim', 100)]
            jobs += [('lin-ddpm-t%d-clip' % t0, 'lin', t0, 'ddpm', None) for t0 in (50, 200, 400)]
            jobs += [('lin-ddim10-t%d-clip' % t0, 'lin', t0, 'ddim', 10) for t0 in (200, 400)]
            for cfg, k, t0, smp, S in jobs:
                if (cfg, seed, b) in done:
                    continue
                sync(); t = time.time()
                if k == 'noise' and smp == 'ddpm':
                    g = gen(seed, b, 'ddpm-full'); out = ddpm(randn(x.shape, g), y, 999, g, clip=True); nfe = 1000
                elif k == 'noise':
                    g = gen(seed, b, 'ddim-full', S); out = ddim(randn(x.shape, g), y, ddim_ts(999, S), clip=True); nfe = S
                elif smp == 'ddpm':
                    g = gen(seed, b, 'ddpm', t0); out = ddpm(warm(k, t0, g), y, t0, g, clip=True); nfe = t0 + 1
                else:
                    g = gen(seed, b, 'ddim', t0, S); out = ddim(warm(k, t0, g), y, ddim_ts(t0, S), clip=True); nfe = S
                sync()
                record(cfg, seed, b, items, out, x, nfe, time.time() - t, k, t0, (smp if S is None else 'ddim%d' % S) + '-clip')
            continue
        # ---- truncated DDIM ----
        for t0, S in DDIM_TR:
            for k in SWEEP_INITS:
                cfg = '%s-ddim%d-t%d' % (k, S, t0)
                if (cfg, seed, b) in done:
                    continue
                g = gen(seed, b, 'ddim', t0, S)
                sync(); t = time.time()
                out = ddim(warm(k, t0, g), y, ddim_ts(t0, S))
                sync()
                record(cfg, seed, b, items, out, x, S + est_nfe[k], time.time() - t + est_sec[k], k, t0, 'ddim%d' % S)
        # ---- full DDIM ----
        for S in FULL_DDIM:
            cfg = 'full-ddim%d' % S
            if (cfg, seed, b) in done:
                continue
            g = gen(seed, b, 'ddim-full', S)
            sync(); t = time.time()
            out = ddim(randn(x.shape, g), y, ddim_ts(999, S))
            sync()
            record(cfg, seed, b, items, out, x, S, time.time() - t, 'noise', 999, 'ddim%d' % S)
        # ---- truncated DDPM ----
        for t0 in DDPM_T0:
            for k in SWEEP_INITS:
                cfg = '%s-ddpm-t%d' % (k, t0)
                if (cfg, seed, b) in done:
                    continue
                g = gen(seed, b, 'ddpm', t0)
                sync(); t = time.time()
                out = ddpm(warm(k, t0, g), y, t0, g)
                sync()
                record(cfg, seed, b, items, out, x, t0 + 1 + est_nfe[k], time.time() - t + est_sec[k], k, t0, 'ddpm')
        # ---- full DDPM ----
        cfg = 'full-ddpm'
        if RUN_FULL_DDPM and (cfg, seed, b) not in done:
            g = gen(seed, b, 'ddpm-full')
            sync(); t = time.time()
            out = ddpm(randn(x.shape, g), y, 999, g)
            sync()
            record(cfg, seed, b, items, out, x, 1000, time.time() - t, 'noise', 999, 'ddpm')
print('finished in %.0fs' % (time.time() - T_START), flush=True)
