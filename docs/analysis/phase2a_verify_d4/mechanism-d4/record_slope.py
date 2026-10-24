"""(1) Cube-level heteroscedasticity from the d = 4 and d = 8 records (no model calls), same estimator as
docs/analysis/phase2a_verify_d8/condition-blindness/record_analysis.py and steelman/record_stats.py:
spread_c = sqrt(spread_ratio_c) * RMSE(mean)_c ; slope of log spread on log RMSE(regressor) and on log RMSE(mean)."""
import json, math, os, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.paired_stats import cluster_se, source_image

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
RECS = {'d4': os.path.join(R, 'uq/d4', 'ev' + 'al', 'd4_contrast.json'),
        'd8': os.path.join(R, 'uq/d8', 'ev' + 'al', 'd8_main.json')}


def boot_slope(x, y, files, n=2000, seed=0):
    """cluster bootstrap (by source class/image cluster as used by cluster_se: the file's parent+image id) of the OLS slope"""
    rng = np.random.default_rng(seed)
    keys = np.array([source_image(f) for f in files])
    uk = np.unique(keys); groups = [np.where(keys == k)[0] for k in uk]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        ii = np.concatenate([groups[p] for p in pick])
        out.append(np.polyfit(x[ii], y[ii], 1)[0])
    return float(np.std(out)), len(uk)


res = {}
for tag, path in RECS.items():
    rec = json.load(open(path))
    pc, files = rec['per_cube'], rec['files']
    rm_reg = np.array(pc['regressor']['rmse_pct']) / 100
    print(f'\n===== {tag}: n {rec["n"]} K {rec["K"]} sigma_r {rec["sigma_r"]:.5f} methods {list(pc)}')
    res[tag] = {}
    for m in ('diffusion', 'varhead', 'b', 'ensemble', 'diffusion_halfcentre'):
        if m not in pc or 'spread_ratio' not in pc[m]:
            continue
        sr = np.array(pc[m]['spread_ratio'])
        rm = np.array(pc[m]['mean_rmse_pct']) / 100
        spread = np.sqrt(sr) * rm
        lx, ly = np.log(rm), np.log(spread)
        slope = np.polyfit(lx, ly, 1)[0]
        slope_reg = np.polyfit(np.log(rm_reg), ly, 1)[0]
        se_b, G = boot_slope(lx, ly, files)
        r = dict(spread_mean_pct=100 * spread.mean(), spread_cov=spread.std() / spread.mean(), err_mean_pct=100 * rm.mean(),
                 err_cov=rm.std() / rm.mean(), slope=slope, slope_se_clusterboot=se_b, slope_on_reg_rmse=slope_reg,
                 pearson_loglog=pearsonr(lx, ly)[0], spearman_spread_err=spearmanr(spread, rm).correlation,
                 corr_logsr_logrmse=np.corrcoef(np.log(sr), lx)[0, 1], sr_p5=np.quantile(sr, .05), sr_p95=np.quantile(sr, .95),
                 spread_p5_pct=100 * np.quantile(spread, .05), spread_p95_pct=100 * np.quantile(spread, .95),
                 err_p5_pct=100 * np.quantile(rm, .05), err_p95_pct=100 * np.quantile(rm, .95))
        res[tag][m] = r
        print(f'{m:22s} slope {slope:.3f} (cluster-boot SE {se_b:.3f}, G {G}) slope-on-reg {slope_reg:.3f} | spread {r["spread_mean_pct"]:.3f}% '
              f'CoV {r["spread_cov"]:.3f} (p5-p95 {r["spread_p5_pct"]:.2f}-{r["spread_p95_pct"]:.2f}) | err {r["err_mean_pct"]:.3f}% CoV {r["err_cov"]:.3f} '
              f'(p5-p95 {r["err_p5_pct"]:.2f}-{r["err_p95_pct"]:.2f}) | Spearman {r["spearman_spread_err"]:.3f} | corr(log sr, log rmse) {r["corr_logsr_logrmse"]:.3f} | sr p5-p95 {r["sr_p5"]:.2f}-{r["sr_p95"]:.2f}')
    # per class coverage / spread ratio for the diffusion and varhead
    cl = np.array(rec['classes'])
    for c in sorted(set(cl)):
        mk = cl == c
        print(f'  class {c} n {mk.sum():3d}: diff cov50/90 {np.mean(np.array(pc["diffusion"]["cov50"])[mk]):.3f}/{np.mean(np.array(pc["diffusion"]["cov90"])[mk]):.3f} '
              f'sr {np.mean(np.array(pc["diffusion"]["spread_ratio"])[mk]):.2f} | vh cov50/90 {np.mean(np.array(pc["varhead"]["cov50"])[mk]):.3f}/{np.mean(np.array(pc["varhead"]["cov90"])[mk]):.3f} '
              f'sr {np.mean(np.array(pc["varhead"]["spread_ratio"])[mk]):.2f} | reg rmse {100*np.mean(rm_reg[mk]):.3f}')
    # S1 mechanics: mean-of-K predicted from x_r + independent noise
    reg = np.array(pc['regressor']['rmse_pct']); single = np.array(pc['diffusion']['single_rmse_pct']); meanK = np.array(pc['diffusion']['mean_rmse_pct'])
    pred = np.sqrt(reg ** 2 + (single ** 2 - reg ** 2) / rec['K'])
    print(f'  mean-of-K RMSE observed {meanK.mean():.4f} vs x_r+independent-noise prediction {pred.mean():.4f}; per-cube Pearson {pearsonr(meanK, pred)[0]:.4f}')
    res[tag]['meanK_obs'] = meanK.mean(); res[tag]['meanK_pred_indep'] = pred.mean()
    # CRPS ratio diffusion / varhead per cube; fraction diffusion better
    cd, cv = np.array(pc['diffusion']['crps']), np.array(pc['varhead']['crps'])
    se_c, G = cluster_se(cd - cv, files)
    print(f'  C1 diff - vh {np.mean(cd - cv):+.4f} se_cl {se_c:.4f}; relative gap {np.mean(cd / cv - 1):.3f}; diffusion better in {np.mean(cd < cv):.3f} of cubes; AUSE diff {np.mean(pc["diffusion"]["ause"]):.3f} vh {np.mean(pc["varhead"]["ause"]):.3f}')
    res[tag]['C1'] = float(np.mean(cd - cv)); res[tag]['C1_rel'] = float(np.mean(cd / cv - 1)); res[tag]['frac_diff_better'] = float(np.mean(cd < cv))
    hs, hr = np.array(pc['diffusion']['single_hf']), np.array(pc['regressor']['hf'])
    print(f'  single-sample hf {hs.mean():+.3f} (e^ = {math.exp(hs.mean()):.2f}x truth), over-sharpened in {np.mean(hs > 0):.3f}; regressor hf {hr.mean():+.3f}; mean-of-K hf {np.mean(pc["diffusion"]["mean_hf"]):+.3f}')
    res[tag]['hf_single'] = float(hs.mean()); res[tag]['hf_reg'] = float(hr.mean())
json.dump(res, open('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/mechanism-d4/record_slope.json', 'w'), indent=1, default=float)
