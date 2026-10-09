"""Write tests/data/baseline_phase0.json, the Phase 0 baseline record.

Regenerate the inputs first (one per d, a few minutes each on CPU), then run this, from the repo root:
    python scripts/eval_warmstart.py --d <d> --baseline          for d in 1 2 4 8
    python scripts/make_baseline_record.py

For each d it records
  - from results/residual_warmstart/eval/baseline_d<d>.json (the loader's 1000 validation files): rmse_pct,
    rmse_pct_se, psnr, sam_deg, rmse_pct_P022 / _P023 / _P024, n;
  - recomputed with baseline_eval: subset40_rmse_pct (the first 40 files of tests/data/val_subset_200.json, pinned
    by tests/test_eval_harness.py), subset200_rmse_pct and subset200_rmse_pct_se (the 200-cube subset the
    diffusion samplers are scored on) and ddpm50_rmse_pct (its 50-cube `ddpm1000_subset`);
  - provenance: prior_s, sigma_d and prior_sha256 (sha256 of the prior npz's bytes).
The recomputed per-cube values are cross-checked against baseline_d<d>.json, so a stale baseline file (written
with another prior) is refused instead of recorded.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.linear_estimate import load_prior  # noqa: E402
from scripts.eval_warmstart import DATA_ROOT, EVAL_DIR, PRIOR, SUBSET, baseline_eval, file_sha256  # noqa: E402

OUT = os.path.join(REPO, 'tests', 'data', 'baseline_phase0.json')
D_VALUES = (1, 2, 4, 8)


def subset_baseline(data_root, prior_path, d, rel_files, full, atol=1e-6):
    """baseline_eval on `rel_files`; every per-cube RMSE must match the 1000-file record `full` within atol."""
    out = baseline_eval(data_root, prior_path, d, [os.path.join(data_root, f) for f in rel_files])
    recorded = dict(zip(full['files'], full['per_item']['rmse_pct']))
    gap = np.abs(np.array([recorded[f] for f in out['files']]) - np.array(out['per_item']['rmse_pct'])).max()
    if gap > atol:
        raise RuntimeError(f'baseline_d{d}.json disagrees with a fresh baseline_eval by {gap:.2e} RMSE points; '
                           f'regenerate it with scripts/eval_warmstart.py --d {d} --baseline')
    return out['mean']


def make_record(data_root=DATA_ROOT, prior_path=PRIOR, eval_dir=EVAL_DIR, d_values=D_VALUES):
    prior = load_prior(prior_path)
    sigma = dict(zip((int(v) for v in prior['d_values']), (float(v) for v in prior['sigma_d'])))
    sha = file_sha256(prior_path)
    with open(SUBSET) as f:
        sub = json.load(f)
    rec = {}
    for d in d_values:
        with open(os.path.join(eval_dir, f'baseline_d{d}.json')) as f:
            full = json.load(f)
        m = full['mean']
        s40 = subset_baseline(data_root, prior_path, d, sub['files'][:40], full)
        s200 = subset_baseline(data_root, prior_path, d, sub['files'], full)
        s50 = subset_baseline(data_root, prior_path, d, sub['ddpm1000_subset'], full)
        rec[f'd{d}'] = dict(
            rmse_pct=round(m['rmse_pct'], 4), rmse_pct_se=round(m['rmse_pct_se'], 5),
            psnr=round(m['psnr'], 3), sam_deg=round(m['sam_deg'], 4),
            rmse_pct_P022=round(m['rmse_pct_P022'], 4), rmse_pct_P023=round(m['rmse_pct_P023'], 4),
            rmse_pct_P024=round(m['rmse_pct_P024'], 4), n=int(full['n']),
            subset40_rmse_pct=round(s40['rmse_pct'], 4),
            subset200_rmse_pct=round(s200['rmse_pct'], 4), subset200_rmse_pct_se=round(s200['rmse_pct_se'], 5),
            ddpm50_rmse_pct=round(s50['rmse_pct'], 4),
            prior_s=float(prior['s']), sigma_d=sigma[d], prior_sha256=sha)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--prior', default=PRIOR)
    args = ap.parse_args()
    rec = make_record(args.data_root, args.prior)
    with open(args.out, 'w') as f:
        json.dump(rec, f, indent=1)
        f.write('\n')
    print(json.dumps(rec, indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
