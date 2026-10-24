"""Finite-K noise floor of S1 on all 200 gate cubes, from the record alone."""
import json, sys
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.paired_stats import cluster_se
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
pc = rec['per_cube']; files = rec['files']; K = rec['K']
for cen, dif in (('regressor', 'diffusion'), ('regressor_halves', 'diffusion_halfcentre')):
    r = np.array(pc[cen]['rmse_pct']); mk = np.array(pc[dif]['mean_rmse_pct']); sg = np.array(pc[dif]['single_rmse_pct'])
    sr = np.array(pc[dif]['spread_ratio'])
    v1 = sr * mk ** 2                                  # mean per-element sample variance (pct^2), ddof=1
    v2 = (sg ** 2 - mk ** 2) * K / (K - 1)             # from the seed-0 draw: E MSE(single) - E MSE(meanK) = v (1 - 1/K)
    d = mk - r
    for name, v in (('spread_ratio', v1), ('single-draw', v2)):
        pred = np.sqrt(r ** 2 + v / K) - r
        resid = d - pred
        se_c, G = cluster_se(resid, files)
        print(f"{dif} vs {cen} [{name}]: mean v {v.mean():.3f} pct^2 (s {np.sqrt(v.mean()):.3f} pct); observed S1 {d.mean():+.4f}; "
              f"predicted finite-K inflation {pred.mean():+.4f}; observed - predicted {resid.mean():+.4f} +- {se_c:.4f} (cluster, G={G}); "
              f"|resid| <= 2 SE: {abs(resid.mean()) <= 2 * se_c}")
    # simple formula from the task: s^2/(2 K rmse)
    print(f"   task formula s^2/(2*K*rmse) with means: {v1.mean() / (2 * K * r.mean()):+.4f}")
    # implied MSE of the sampler mean mu vs x_r: MSE(mu) = E MSE(m) - v/K
    mse_mu = mk ** 2 - v1 / K
    print(f"   implied RMSE of the sampler expectation mu (meanK^2 - v/K): {np.sqrt(np.maximum(mse_mu, 0)).mean():.4f} vs regressor {r.mean():.4f}; "
          f"diff {np.mean(np.sqrt(np.maximum(mse_mu, 0)) - r):+.4f}")
    # K needed for the null S1 inflation to fall within 2 SE_cluster of zero
    se_obs = cluster_se(d, files)[0]
    for KK in (32, 64, 128, 256, 512):
        pk = (np.sqrt(r ** 2 + v1 / KK) - r).mean()
        print(f"   K={KK}: null inflation {pk:+.4f} vs 2 SE_cluster {2*se_obs:.4f}")
