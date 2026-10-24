"""Post hoc spread-error summaries for the Phase 2a records (final-review item SE-4; report only, no gate depends on them).

Spec §5.3 asks for the spread ratio and for the per-cube correlation between predicted and actual error energy.
uq_eval stores the per-cube spread ratio  sr_i = mean(pred_var_i) / mean(err_i^2)  (err = sample mean - truth, [0, 1]
scale) and the per-cube RMSE (%) of the same mean, so mean(err_i^2) = (rmse_i / 100)^2 and mean(pred_var_i) = sr_i * that.
This script derives, per record and method:
  - mean_of_ratios: the recorded summary (mean over cubes of sr_i);
  - ratio_of_means: sum_i pred_var_i / sum_i err_i^2 (the pooled calibration of the spread's size);
  - ratio_of_means_kc (diffusion only): the same with the mean-of-K's Monte-Carlo term removed from the denominator,
    sum_i pred_var_i / sum_i (err_i^2 - pred_var_i / K). The mean of K draws has E[err^2] = (centre error)^2 + v / K, so
    this compares the spread with the error of the sampler's centre and is comparable across K (at K = 32 it is about
    2 % above ratio_of_means, at K = 8 about 14 %). The variance head and the ensemble have no such term;
  - pearson_log: Pearson correlation over cubes of log pred_var_i with log err_i^2 (does the spread track the error);
  - spearman: rank correlation of the same two quantities;
with 95 % cluster-bootstrap intervals that resample source images (2000 draws, seed 0).
The RMSE is stored in float32, so pred_var is reconstructed to about 1e-7 relative error; that is far below any
reported digit. Reads the records read-only; writes spread_error.json next to this file. Run from anywhere:

    python docs/analysis/phase2a_page/spread_error_posthoc.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from scripts.paired_stats import source_image  # noqa: E402

R = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
RECORDS = {'d8': 'uq/d8/eval/d8_main.json', 'd8k64': 'uq/d8/eval/d8_fix_k64.json',
           'd4': 'uq/d4/eval/d4_contrast.json', 'd8ddpm': 'uq/d8/eval/d8_fix_ddpm.json'}
METHODS = ('diffusion', 'varhead', 'ensemble')
B, SEED = 2000, 0


def stats(pv, e2, K=None):
    lp, le = np.log(pv), np.log(e2)
    rp = np.argsort(np.argsort(pv)).astype(float)
    re_ = np.argsort(np.argsort(e2)).astype(float)
    out = dict(ratio_of_means=float(pv.sum() / e2.sum()), pearson_log=float(np.corrcoef(lp, le)[0, 1]),
               spearman=float(np.corrcoef(rp, re_)[0, 1]))
    if K:
        out['ratio_of_means_kc'] = float(pv.sum() / (e2 - pv / K).sum())
    return out


def summarise(rec, method, rng):
    pc = rec['per_cube'][method]
    sr = np.asarray(pc['spread_ratio'], dtype=np.float64)
    e2 = (np.asarray(pc['mean_rmse_pct'], dtype=np.float64) / 100.0) ** 2
    pv = sr * e2
    K = rec['K'] if method == 'diffusion' else None
    point = stats(pv, e2, K)
    keys = np.array([source_image(f) for f in rec['files']])
    groups = [np.flatnonzero(keys == k) for k in np.unique(keys)]
    boot = {k: [] for k in point}
    for _ in range(B):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for k, v in stats(pv[idx], e2[idx], K).items():
            boot[k].append(v)
    out = dict(mean_of_ratios=float(sr.mean()), n=int(len(sr)), clusters=len(groups), K=K)
    for k, v in point.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out[k] = dict(value=v, ci95=[float(lo), float(hi)])
    return out


def main():
    res = {}
    for key, rp in RECORDS.items():
        p = os.path.join(R, rp)
        if not os.path.exists(p):
            res[key] = None
            continue
        with open(p) as f:
            rec = json.load(f)
        rng = np.random.default_rng(SEED)
        res[key] = {m: summarise(rec, m, rng) for m in METHODS if m in rec['per_cube']}
        res[key]['record'] = rp
    out = os.path.join(HERE, 'spread_error.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    for key, v in res.items():
        if not v:
            print(f'{key}: no record')
            continue
        for m in METHODS:
            if m not in v:
                continue
            s = v[m]
            print(f"{key:7s} {m:9s} n={s['n']:3d} mean-of-ratios {s['mean_of_ratios']:.3f}  ratio-of-means {s['ratio_of_means']['value']:.3f} "
                  f"[{s['ratio_of_means']['ci95'][0]:.3f}, {s['ratio_of_means']['ci95'][1]:.3f}]  r(log) {s['pearson_log']['value']:.3f} "
                  f"[{s['pearson_log']['ci95'][0]:.3f}, {s['pearson_log']['ci95'][1]:.3f}]  rho {s['spearman']['value']:.3f}"
                  + (f"  K-corrected ratio-of-means {s['ratio_of_means_kc']['value']:.3f} [{s['ratio_of_means_kc']['ci95'][0]:.3f}, {s['ratio_of_means_kc']['ci95'][1]:.3f}]" if 'ratio_of_means_kc' in s else ''))
    print('wrote', out)


if __name__ == '__main__':
    main()
