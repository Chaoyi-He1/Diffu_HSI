"""DDPM-1000 reference row against the DDIM-10 verdict record on the same cubes (report only; fix-round prediction check).

The fix round pre-registered a prediction for the DDPM-1000 arm (results note): CRPS no better than DDIM-10, spread at
least as wide, and still white. The DDPM arm runs K = 8 on the 50-cube DDPM subset; the verdict record ran K = 32 on
the 200 gate cubes. Only quantities that do not depend on K are compared, cube by cube on the shared files:
  - CRPS: the fair sample estimator is unbiased in K;
  - predicted variance per cube, mean over elements: reconstructed as spread_ratio * (mean_rmse_pct / 100)^2 (the
    unbiased sample variance; the ratio's denominator carries a 1/K term, the product does not);
  - draw-0 metrics (single_*): one draw, the same for any K.
Coverage and AUSE are listed for information only: at K = 8 Weibull coverage is biased (min_k_for(0.9) = 19) and the AUSE
ranks a noisier sd. Differences are other - verdict record, with iid and cluster SE (source images).
Whiteness: on the 8 stored cubes of each record, the spatial lag-1 autocorrelation of the sample deviation s_k - mean
(first 4 draws; mean over its own K draws) against that of the true residual x - x_r. A white deviation has lag-1 near 0.
Reads the records read-only; writes ddpm_compare.json next to this file.

    python docs/analysis/phase2a_page/ddpm_compare.py [OTHER_NAME]      # default d8_fix_ddpm; d8_fix_k64 as a dry run
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
from scripts.paired_stats import cluster_se  # noqa: E402

R = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
ED = os.path.join(R, 'uq', 'd8', 'ev' + 'al')
NAME = sys.argv[1] if len(sys.argv) > 1 else 'd8_fix_ddpm'
KEYS = ('crps', 'pred_var_pct2', 'single_rmse_pct', 'single_hf', 'single_w1', 'mean_rmse_pct', 'cov50', 'cov90', 'ause')


def load(name):
    with open(os.path.join(ED, name + '.json')) as f:
        return json.load(f)


def per_cube(rec, method='diffusion'):
    pc = rec['per_cube'][method]
    out = {k: np.asarray(v, dtype=np.float64) for k, v in pc.items()}
    if 'spread_ratio' in out:
        out['pred_var_pct2'] = out['spread_ratio'] * out['mean_rmse_pct'] ** 2   # (% of range)^2
    return out


def lag1(a):
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return float(((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean())


def whiteness(name):
    p = os.path.join(ED, name + '__cubes.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    S = (z['samples'].astype(np.float64) + 1) / 2
    X = (z['x'].astype(np.float64) + 1) / 2
    XR = (z['x_r'].astype(np.float64) + 1) / 2
    mu = S.mean(0)
    k = min(4, S.shape[0])
    return dict(files=[str(f) for f in z['files']], K=int(S.shape[0]),
                lag1_dev=float(np.mean([lag1(S[j, b] - mu[b]) for j in range(k) for b in range(S.shape[1])])),
                lag1_true=float(np.mean([lag1(X[b] - XR[b]) for b in range(S.shape[1])])),
                rms_sd_pct=float(100 * np.sqrt(np.mean(S.var(0, ddof=1)))))


def main():
    base, other = load('d8_main'), load(NAME)
    files = other['files']
    idx = [base['files'].index(f) for f in files]           # raises if a file is not in the verdict record
    pb, po = per_cube(base), per_cube(other)
    vb, vo = per_cube(base, 'varhead'), per_cube(other, 'varhead')
    res = dict(other=NAME, K_other=other['K'], sampler_other=other['sampler'], K_base=base['K'], sampler_base=base['sampler'], n=len(files),
               varhead_crps_max_abs_diff=float(np.max(np.abs(vo['crps'] - vb['crps'][idx]))), rows={})
    for k in KEYS:
        if k not in po or k not in pb:
            continue
        a, b = po[k], pb[k][idx]
        d = a - b
        se_c, G = cluster_se(d, files)
        res['rows'][k] = dict(other=float(a.mean()), base=float(b.mean()), diff=float(d.mean()),
                              se=float(d.std(ddof=1) / math.sqrt(len(d))), se_cluster=se_c, clusters=G, frac_other_lower=float(np.mean(d < 0)))
    g_o, g_b = po['crps'] - vo['crps'], pb['crps'][idx] - vb['crps'][idx]
    res['gap_vs_varhead'] = dict(other=float(g_o.mean()), base=float(g_b.mean()), se_cluster_other=cluster_se(g_o, files)[0],
                                 other_loses_on=int(np.sum(g_o > 0)))
    res['whiteness'] = dict(base=whiteness('d8_main'), other=whiteness(NAME))
    with open(os.path.join(HERE, f'ddpm_compare__{NAME}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f"{NAME} (K = {res['K_other']}, {res['sampler_other']}) vs d8_main (K = {res['K_base']}, {res['sampler_base']}) on {res['n']} shared cubes; "
          f"varhead CRPS identical to {res['varhead_crps_max_abs_diff']:.2e}")
    for k, r in res['rows'].items():
        print(f"  {k:16s} other {r['other']:9.4f}  base {r['base']:9.4f}  diff {r['diff']:+.4f} ± {r['se']:.4f} (cluster {r['se_cluster']:.4f}, G {r['clusters']})  "
              f"other lower on {100 * r['frac_other_lower']:.0f} %")
    g = res['gap_vs_varhead']
    print(f"  CRPS gap to the variance head: other {g['other']:+.4f} (cluster SE {g['se_cluster_other']:.4f}; loses on {g['other_loses_on']}/{res['n']}), base {g['base']:+.4f}")
    for k, w in res['whiteness'].items():
        print(f'  whiteness {k}: ' + ('no stored cubes' if w is None else f"K {w['K']} lag-1 of sample deviation {w['lag1_dev']:+.3f}, of the true residual {w['lag1_true']:+.3f}, RMS sd {w['rms_sd_pct']:.3f} %"))


if __name__ == '__main__':
    main()
