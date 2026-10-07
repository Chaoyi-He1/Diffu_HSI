"""Step 3 (CPU): aggregate records_<set>.jsonl into results.json + a printed table.
Mean +- std are over (test item x noise seed); 'seed_sd' is the std over seeds of the
per-seed mean; paired differences vs full DDPM and vs the lin estimate use the same
(item, seed) pairs (common random numbers)."""
import os, sys, json
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__))


def load(setname):
    p = os.path.join(OUT, 'records_%s.jsonl' % setname)
    if not os.path.exists(p):
        return None
    recs = {}
    for line in open(p):
        r = json.loads(line)
        recs[(r['config'], r['seed'], r['batch'])] = r
    by = {}
    for (cfg, seed, b), r in recs.items():
        d = by.setdefault(cfg, dict(init=r['init'], t0=r['t0'], sampler=r['sampler'], nfe=r['nfe'], cells={}, sec=[]))
        for j, it in enumerate(r['items']):
            d['cells'][(seed, it)] = (r['rmse'][j], r['sam'][j], r['psnr'][j])
        d['sec'].append(r['sec_per_item'])
    return by


def order_key(cfg, d):
    s = d['sampler']; i = d['init']
    grp = 0 if cfg.startswith('full') else (1 if cfg.startswith('est') else (2 if s == 'ddpm' else 3))
    return (grp, ['noise', 'lin', 'gauss', 'self', 'mean', 'linreg'].index(i), -(d['t0'] or 0), d['nfe'])


def summarize(setname):
    by = load(setname)
    if not by:
        return None
    ref_cells = by.get('full-ddpm', {}).get('cells', {})
    lin_cells = by.get('est-lin', {}).get('cells', {})
    # per-item sampling noise of full DDPM (two seeds of the same item)
    noise = None
    if ref_cells:
        seeds = sorted({s for s, _ in ref_cells}); items = sorted({i for _, i in ref_cells})
        if len(seeds) >= 2:
            sd = [np.std([ref_cells[(s, i)][0] for s in seeds if (s, i) in ref_cells], ddof=1)
                  for i in items if sum((s, i) in ref_cells for s in seeds) >= 2]
            noise = float(np.mean(sd))
    rows = []
    for cfg, d in sorted(by.items(), key=lambda kv: order_key(*kv)):
        keys = sorted(d['cells'])
        a = np.array([d['cells'][k] for k in keys])           # [n, 3]
        seeds = sorted({s for s, _ in keys})
        seed_means = [a[[k[0] == s for k in keys], 0].mean() for s in seeds]
        row = dict(config=cfg, init=d['init'], t0=d['t0'] if d['t0'] is not None else 0, sampler=d['sampler'],
                   nfe=d['nfe'], n=len(keys), seeds=len(seeds),
                   rmse_pct=float(a[:, 0].mean()), rmse_std=float(a[:, 0].std(ddof=1)) if len(a) > 1 else 0.0,
                   rmse_seed_sd=float(np.std(seed_means, ddof=1)) if len(seeds) > 1 else None,
                   sam_deg=float(a[:, 1].mean()), sam_std=float(a[:, 1].std(ddof=1)) if len(a) > 1 else 0.0,
                   psnr=float(a[:, 2].mean()), psnr_std=float(a[:, 2].std(ddof=1)) if len(a) > 1 else 0.0,
                   sec_per_item=float(np.mean(d['sec'])))
        for name, ref in (('vs_full_ddpm', ref_cells), ('vs_lin', lin_cells)):
            common = [k for k in keys if k in ref]
            if common and cfg not in ('full-ddpm',) and ref is not d['cells']:
                diff = np.array([d['cells'][k][0] - ref[k][0] for k in common])
                dsam = np.array([d['cells'][k][1] - ref[k][1] for k in common])
                row[name] = dict(d_rmse=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                 d_sam=float(dsam.mean()), se_sam=float(dsam.std(ddof=1) / np.sqrt(len(dsam))),
                                 frac_better=float((diff < 0).mean()), n=len(common))
        own = by.get('est-%s' % d['init'], {}).get('cells') if not cfg.startswith('est') else None
        if own:
            common = [k for k in keys if k in own]
            diff = np.array([d['cells'][k][0] - own[k][0] for k in common])
            row['vs_own_estimate'] = dict(d_rmse=float(diff.mean()), se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                          frac_better=float((diff < 0).mean()), n=len(common))
        rows.append(row)
    min_cell = None
    if lin_cells:
        min_cell = min((min(d['cells'][k][0] - lin_cells[k][0] for k in d['cells'] if k in lin_cells), c)
                       for c, d in by.items() if c != 'est-lin')
    return dict(rows=rows, full_ddpm_per_item_seed_sd_rmse=noise,
                smallest_cellwise_rmse_gap_to_lin=dict(config=min_cell[1], d_rmse=float(min_cell[0])) if min_cell else None)


def fmt(rows, title):
    print('\n### %s' % title)
    print('%-22s %-6s %4s %-7s %5s  %-15s %-15s %-14s %8s  %-22s %-22s' % (
        'config', 'init', 't0', 'sampler', 'NFE', 'RMSE % (sd)', 'SAM deg (sd)', 'PSNR dB', 's/item',
        'dRMSE vs full (se)', 'dRMSE vs lin (se)'))
    for r in rows:
        vf = r.get('vs_full_ddpm'); vl = r.get('vs_lin')
        print('%-22s %-6s %4s %-7s %5d  %6.3f (%5.3f)  %6.3f (%5.3f)  %6.2f (%4.2f) %8.3f  %-22s %-22s' % (
            r['config'], r['init'], r['t0'], r['sampler'], r['nfe'], r['rmse_pct'], r['rmse_std'],
            r['sam_deg'], r['sam_std'], r['psnr'], r['psnr_std'], r['sec_per_item'],
            ('%+.3f (%.3f)' % (vf['d_rmse'], vf['se'])) if vf else '-',
            ('%+.3f (%.3f)' % (vl['d_rmse'], vl['se'])) if vl else '-'))


if __name__ == '__main__':
    res = {}
    for s in ('primary', 'secondary'):
        r = summarize(s)
        if r:
            res[s] = r
            fmt(r['rows'], s + ('  (full-DDPM per-item seed sd of RMSE: %.3f)' % r['full_ddpm_per_item_seed_sd_rmse']
                                if r['full_ddpm_per_item_seed_sd_rmse'] is not None else ''))
    meta_p = os.path.join(OUT, 'testsets.json')
    if os.path.exists(meta_p):
        m = json.load(open(meta_p))
        res['setup'] = dict(lin_regulariser_s=m['s'], s_grid_heldout_rmse=m['s_grid'], prior_fit_files=m['fit_files'],
                            prior_fit_pixels=m['fit_pixels'], s_selection_files=m['val_files'],
                            primary_files=m['primary'], secondary_files=m['secondary'])
    res.setdefault('setup', {}).update(dict(
        checkpoint='/data/chaoyi_he/HSI/Diffu/results/2d_hsi_diffusion/HFD/R_1/l2_loss/checkpoint_epoch_101.pth',
        code_snapshot='988a338 (strict=True load OK)', model='U2NetHyperspectral(64, 30, base_channels=128), eps-prediction',
        schedule='linear betas 1e-4..0.02, T=1000', linreg_s=0.1, self_estimate='5-step DDIM t=999,799,599,400,200 -> 0',
        precision='fp16 autocast + fused SDPA attention, batch 8, GPU 0 capped at 2.4 GiB',
        metrics='outputs clamped to [-1,1]; RMSE % and PSNR on (x+1)/2; SAM deg on (x+1)/2; mean/sd over 16 items x 2 seeds'))
    dp = os.path.join(OUT, 'diag_denoise.json')
    if os.path.exists(dp):
        res['diagnostics'] = dict(one_step_denoising=json.load(open(dp)),
                                  note='x0_rmse: one-step x0 estimate from x_t = q(x_t|x_true) with true y; '
                                       'x0_rmse_yshuf: same with another item\'s y; x0_rmse_y0: with y = 0')
    sp = os.path.join(OUT, 'diag_samples.log')
    if os.path.exists(sp):
        res.setdefault('diagnostics', {})['pure_noise_samples_log'] = open(sp).read()
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(OUT, 'results.json'))
