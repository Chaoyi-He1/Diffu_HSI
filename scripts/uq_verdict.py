"""The pre-registered Phase 2a verdict (spec §5.7) read from one uq_eval record. Written before any Stage 1 number
exists; the thresholds live here and nowhere else.

    python scripts/uq_verdict.py $R/uq/d8/eval/d8_main.json [--no_centre_check] [--report_only]

Calibration passes = C1 and C2; perceptual passes = P1 and P2; "diffusion wins (a)" = both and S1 and S2. The d = 4
contrast runs with --report_only (same clauses, no gate). The record's centre must be the pre-registered selected
regressor of its stride unless --no_centre_check (smoke tests); either flag is spelled into the verdict file name.
C2's empirical coverage is exact only for K >= min_k_for(0.9) = 19 (R10): a record below that is refused unless
--report_only, which prints a warning instead.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.warmstart_paths import UQ_DIR, git_head  # noqa: E402
from scripts.paired_stats import cluster_se  # noqa: E402
from scripts.uq.metrics import min_k_for  # noqa: E402
from scripts.uq_eval import PREREG_CENTRE  # noqa: E402

COV90, COV50 = (0.85, 0.95), (0.45, 0.55)
NEEDED = ('diffusion', 'varhead', 'regressor', 'baseline')


def paired(a, b, files):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    se_c, G = cluster_se(d, files)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(len(d))), se_cluster=se_c, n=int(len(d)), clusters=G)


def _m(v):
    return float(np.mean(v))


def clauses(rec):
    pc, files = rec['per_cube'], rec['files']
    missing = [m for m in NEEDED if m not in pc]
    if missing:
        raise ValueError(f'record {rec.get("name")} lacks the methods {missing}; the verdict needs {NEEDED}')
    out = {}
    c1 = paired(pc['diffusion']['crps'], pc['varhead']['crps'], files)
    out['C1'] = dict(c1, rule='mean(CRPS_diffusion - CRPS_varhead) + 2 SE_cluster < 0', passed=bool(c1['mean'] + 2 * c1['se_cluster'] < 0))
    cov90, cov50 = _m(pc['diffusion']['cov90']), _m(pc['diffusion']['cov50'])
    out['C2'] = dict(cov90=cov90, cov50=cov50, rule=f'{COV90[0]} <= cov90 <= {COV90[1]} and {COV50[0]} <= cov50 <= {COV50[1]}',
                     passed=bool(COV90[0] <= cov90 <= COV90[1] and COV50[0] <= cov50 <= COV50[1]))
    p1 = paired(np.abs(pc['diffusion']['single_hf']), np.abs(pc['regressor']['hf']), files)
    out['P1'] = dict(p1, rule='mean(|hf_single| - |hf_regressor|) + 2 SE_cluster < 0', passed=bool(p1['mean'] + 2 * p1['se_cluster'] < 0))
    p2 = paired(pc['diffusion']['single_rmse_pct'], pc['baseline']['rmse_pct'], files)
    out['P2'] = dict(p2, rule='mean(RMSE_single - RMSE_baseline) + 2 SE_cluster < 0', passed=bool(p2['mean'] + 2 * p2['se_cluster'] < 0))
    s1 = paired(pc['diffusion']['mean_rmse_pct'], pc['regressor']['rmse_pct'], files)
    out['S1'] = dict(s1, rule='|mean(RMSE_meanK - RMSE_regressor)| <= 2 SE_cluster', passed=bool(abs(s1['mean']) <= 2 * s1['se_cluster']))
    ys = rec.get('yswap') or {}
    if not ys:
        raise ValueError('record has no yswap block (S2 needs the mean-of-K y-swap)')
    d = np.asarray(ys['swap_full'], dtype=np.float64) - np.asarray(ys['true'], dtype=np.float64)
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    out['S2'] = dict(mean=float(d.mean()), se=se, n=int(len(d)), rule='mean(RMSE_swap_full - RMSE_true) - 2 SE_iid > 0 on the y-swap cubes',
                     passed=bool(d.mean() - 2 * se > 0))
    rep = dict(C3=paired(pc['diffusion']['ause'], pc['varhead']['ause'], files),
               spread_ratio=dict(diffusion=_m(pc['diffusion']['spread_ratio']), varhead=_m(pc['varhead']['spread_ratio'])),
               cov90_varhead=_m(pc['varhead']['cov90']), cov90_gauss_diffusion=_m(pc['diffusion'].get('cov90_gauss', [float('nan')])))
    if 'ensemble' in pc:
        rep['ensemble'] = dict(crps=paired(pc['diffusion']['crps'], pc['ensemble']['crps'], files), ause=paired(pc['diffusion']['ause'], pc['ensemble']['ause'], files),
                               cov90=_m(pc['ensemble']['cov90']))
    if 'diffusion_halfcentre' in pc:
        rep['halfcentre'] = dict(crps=paired(pc['diffusion_halfcentre']['crps'], pc['diffusion']['crps'], files),
                                 cov90=_m(pc['diffusion_halfcentre']['cov90']))
    out['reported'] = rep
    out['calibration_pass'] = bool(out['C1']['passed'] and out['C2']['passed'])
    out['perceptual_pass'] = bool(out['P1']['passed'] and out['P2']['passed'])
    out['wins'] = bool(out['calibration_pass'] and out['perceptual_pass'] and out['S1']['passed'] and out['S2']['passed'])
    return out


def check_centre(rec):
    want = PREREG_CENTRE.get(int(rec['d']))
    c = (rec.get('provenance') or {}).get('centre') or {}
    got = (c.get('run'), c.get('epoch'))
    if want is None or got != want:
        raise ValueError(f'record centre {got} is not the pre-registered selected regressor {want} for d = {rec["d"]} '
                         f'(use --no_centre_check only for smoke tests)')


def check_k(rec):
    """R10: C2 compares the empirical coverage with 0.85-0.95 / 0.45-0.55, which is meaningful only when the sample
    quantiles are genuine order statistics, i.e. K >= min_k_for(0.9)."""
    need = min_k_for(0.9)
    if not rec.get('coverage_exact', True) or rec['K'] < need:
        raise ValueError(f'record K = {rec["K"]}, coverage_exact = {rec.get("coverage_exact")}: the empirical coverage of C2 is exact only '
                         f'for K >= scripts.uq.metrics.min_k_for(0.9) = {need} (use --report_only for a descriptive verdict)')


def verdict_path(rec, no_centre_check, report_only):
    return os.path.join(UQ_DIR, f'd{rec["d"]}', f'verdict__{rec["name"]}' + ('__nocentre' if no_centre_check else '')
                        + ('__report' if report_only else '') + '.json')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('record'); ap.add_argument('--no_centre_check', action='store_true'); ap.add_argument('--report_only', action='store_true')
    args = ap.parse_args(argv)
    with open(args.record) as f:
        rec = json.load(f)
    if not args.no_centre_check:
        check_centre(rec)
    try:
        check_k(rec)
    except ValueError as e:
        if not args.report_only:
            raise
        print(f'WARNING (report only): {e}')
    out = verdict_path(rec, args.no_centre_check, args.report_only)
    if os.path.exists(out):                       # after the record checks: a refused record is reported as refused, not as taken
        raise FileExistsError(f'{out} exists; the verdict is taken once')
    c = clauses(rec)
    for k in ('C1', 'C2', 'P1', 'P2', 'S1', 'S2'):
        v = c[k]
        stat = f"{v['mean']:+.4f} ± {v['se']:.4f}" + (f" (cluster {v['se_cluster']:.4f})" if 'se_cluster' in v else '') if 'mean' in v else f"cov90 {v['cov90']:.3f} cov50 {v['cov50']:.3f}"
        print(f"{k}: {'PASS' if v['passed'] else 'FAIL'}  {stat}   [{v['rule']}]")
    print(f"calibration {'PASS' if c['calibration_pass'] else 'FAIL'}; perceptual {'PASS' if c['perceptual_pass'] else 'FAIL'}; "
          f"diffusion wins (a): {c['wins']}" + ('   (REPORT ONLY, no gate)' if args.report_only else ''))
    v = dict(c, record=os.path.abspath(args.record), name=rec['name'], d=rec['d'], K=rec['K'], sampler=rec['sampler'], n=rec['n'],
             centre=(rec.get('provenance') or {}).get('centre'), report_only=bool(args.report_only), no_centre_check=bool(args.no_centre_check),
             git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.tmp', 'w') as f:
        json.dump(v, f, indent=1)
    os.replace(out + '.tmp', out)
    print('wrote', out)
    return v


if __name__ == '__main__':
    main()
