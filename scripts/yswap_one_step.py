"""One-step y-swap of one checkpoint at any t, on CPU (final-review fix item 1; a post-hoc diagnostic).

  CUDA_VISIBLE_DEVICES= python scripts/yswap_one_step.py \
      --checkpoint <results>/runs/residual_d4_b64/checkpoint_epoch_100.pth --mode residual \
      --t 100 300 999 --n 32 --out /tmp/yswap_residual_d4_n32.json

B's y-swap for gate_1a_prime (--b_yswap) is the leak-free one at its one-step call:

  CUDA_VISIBLE_DEVICES= python scripts/yswap_one_step.py \
      --checkpoint <results>/runs/residual_d4_b64/checkpoint_epoch_100.pth --mode residual \
      --t 999 --n 32 --x_t noise --out <results>/diagnostics/yswap_residual_d4_b64_t999_n32__xT.json

It builds the harness's stratified y-swap batch of n cubes (eval_warmstart.yswap_batch: yswap_files over the 200-cube
subset; n = 8 is the spec §7 batch, n = 32 the gate's), runs eval_warmstart.yswap_check at each t (x_t fixed by seed
1234 for every variant; every variant composed with the TRUE x0_hat) and adds, per t, the paired per-cube mean +- SE
(iid, and CR1 clustered by source image) of swap_full - true, true - baseline, swap_full - baseline and
swap_sensor_only - true, the ratios true / swap_full and true / baseline, and how many cubes the true condition wins.

--x_t (final-review F1; recorded as provenance.x_t): 'q_sample' (default) feeds x_t = q_sample(true residual, t), as
spec §7 and the Phase 1a records do; at t = 999 that x_t still carries 0.00635 x the true residual, which B reads but
its real one-step read-out never sees (a leak). 'noise' feeds the ddim1 sampler's own pure-noise x_T and is defined
only at t = 999. Mode regress calls on its zero x_t whatever --x_t says (provenance.x_t 'zero').

The spec §7 rule (true <= 0.9 x swap_full at t = 100 and 300) and the gate are unchanged; this script only reports.
It refuses an --out inside the results tree (RESULTS_ROOT) other than RESULTS_ROOT/diagnostics and an --out that
exists (it writes via a temporary file), and checks the checkpoint's meta like the harness does. provenance.git_head
is the HEAD at launch. Mode regress needs --prediction_type direct; its y-swap is the single call at t = 999 (--t is
ignored).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, RESULTS_ROOT, file_sha256, git_head  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import MODES, XR_MODES  # noqa: E402
from scripts.eval_warmstart import (YSWAP_NOISE_SEED, YSWAP_X_T, _mean_se, build_model, check_meta,  # noqa: E402
                                    check_yswap_x_t, yswap_batch, yswap_check)
from scripts.paired_stats import cluster_se  # noqa: E402

DIAG_DIR = os.path.join(RESULTS_ROOT, 'diagnostics')   # the one place in the results tree this script may write

VARIANTS = ('true', 'swap_full', 'swap_sensor_only', 'zero')
CONTRASTS = (('swap_full_minus_true', 'swap_full', 'true'),
             ('true_minus_baseline', 'true', 'baseline'),
             ('swap_full_minus_baseline', 'swap_full', 'baseline'),
             ('swap_sensor_only_minus_true', 'swap_sensor_only', 'true'))


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--mode', required=True, choices=[m for m in MODES if m not in XR_MODES],
                    help='a Phase 1 mode; the Phase 2a modes (residual_reg, varhead) are evaluated with scripts/uq_eval.py')
    ap.add_argument('--t', type=int, nargs='+', default=[100, 300, 999])
    ap.add_argument('--n', type=int, default=32, help='cubes, picked by yswap_files (32 = the gate batch, 8 = spec §7)')
    ap.add_argument('--x_t', choices=list(YSWAP_X_T), default='q_sample',
                    help="the network's x_t: q_sample(true residual, t) (default; leaks the residual at t = 999) or "
                         "'noise', the ddim1 sampler's own x_T (t = 999 only)")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', required=True, help='JSON output; inside the results tree only under RESULTS_ROOT/diagnostics')
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--prediction_type', default='v')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--prior', default=PRIOR)
    return ap


def under(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def summarize_t(res):
    """Per t (str): the four variant means (yswap_check's) and the baseline mean, the ratios true / swap_full and
    true / baseline, and the paired per-cube contrasts with iid and clustered SEs over res['files']."""
    files, base = res['files'], np.asarray(res['baseline_per_cube'], dtype=np.float64)
    out = {}
    for t, cubes in res['per_cube'].items():
        v = {k: np.asarray(cubes[k], dtype=np.float64) for k in VARIANTS}
        v['baseline'] = base
        means = dict(res[f't{t}'], baseline=float(base.mean()))
        paired = {}
        for name, a, b in CONTRASTS:
            d = v[a] - v[b]
            mean, se = _mean_se(d)
            se_cluster, n_clusters = cluster_se(d, files)
            paired[name] = dict(mean=mean, se=se, se_cluster=se_cluster, n_clusters=n_clusters, n=int(len(d)),
                                n_positive=int((d > 0).sum()))
        out[t] = dict(means=means, ratio_true_over_swap_full=means['true'] / means['swap_full'],
                      ratio_true_over_baseline=means['true'] / means['baseline'], paired=paired,
                      n_true_beats_swap_full=paired['swap_full_minus_true']['n_positive'])
    return out


def print_summary(summary, n):
    for t, s in summary.items():
        m = s['means']
        print(f"t={t:>4}  true {m['true']:.4f}  swap_full {m['swap_full']:.4f}  swap_sensor_only {m['swap_sensor_only']:.4f}  "
              f"zero {m['zero']:.4f}  baseline {m['baseline']:.4f}  true/swap_full {s['ratio_true_over_swap_full']:.4f}  "
              f"true wins {s['n_true_beats_swap_full']}/{n}")
        for name, c in s['paired'].items():
            cl = f"  (clustered ± {c['se_cluster']:.4f} over {c['n_clusters']} images)" if c['se_cluster'] is not None else ''
            print(f"        {name:<28} {c['mean']:+.4f} ± {c['se']:.4f}{cl}  positive on {c['n_positive']}/{c['n']}")


def main(argv=None):
    args = get_parser().parse_args(argv)
    head = git_head()                                   # the HEAD at launch (final-review F5), before any work
    if under(args.out, RESULTS_ROOT) and not under(args.out, DIAG_DIR):
        raise SystemExit(f'--out {args.out} is inside the results tree {RESULTS_ROOT}; this diagnostic writes there only '
                         f'under {DIAG_DIR}')
    if os.path.exists(args.out):
        raise SystemExit(f'{args.out} exists; yswap_one_step never overwrites a record (choose another --out)')
    device = torch.device(args.device)
    trainer = DiffusionTrainer(device=device, prediction_type=args.prediction_type, loss_type='l1', snr_gamma=None)
    x_t = 'zero' if args.mode == 'regress' else args.x_t     # the regressor calls on its zero x_t; --x_t does not apply
    if args.mode != 'regress':
        check_yswap_x_t(args.x_t, args.t, trainer.n_timesteps)    # before the checkpoint and the cubes are loaded
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ds, batch = yswap_batch(args.data_root, args.prior, args.d, args.n)
    sigma_d, prior_sha256 = ds.sigma_d, file_sha256(args.prior)
    meta_args = SimpleNamespace(ckpt=args.checkpoint, mode=args.mode, d=args.d, base_channels=args.base_channels,
                                prediction_type=args.prediction_type)
    meta_checked = check_meta(ck, meta_args, sigma_d, prior_sha256)
    model = build_model(args.mode, args.base_channels, device)
    model.load_state_dict(ck.get('model_state_dict', ck), strict=True)
    files = [os.path.relpath(f, args.data_root) for f in ds.img_list]
    t0 = time.time()
    res = yswap_check(model, trainer, args.mode, batch, t_values=tuple(args.t), sigma_d=sigma_d, files=files,
                      x_t=args.x_t)
    secs = time.time() - t0
    summary = summarize_t(res)
    provenance = dict(checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                      mode=args.mode, d=args.d, base_channels=args.base_channels, prediction_type=args.prediction_type,
                      sigma_d=sigma_d, prior_path=os.path.abspath(args.prior), prior_sha256=prior_sha256,
                      meta_checked=meta_checked, device=str(device), x_t=x_t, noise_seed=YSWAP_NOISE_SEED,
                      torch=torch.__version__, git_head=head, argv=list(sys.argv[1:] if argv is None else argv),
                      seconds=secs)
    out = dict(provenance=provenance, n=len(files), t_values=list(args.t), files=files, summary=summary,
               passes_0p9_rule_at_these_t=res['passes'], yswap=res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)
    print(f'{args.mode} {os.path.basename(args.checkpoint)}: one-step y-swap on {len(files)} cubes ({device}, {secs:.0f} s, '
          f'x_t {x_t})')
    print_summary(summary, len(files))
    print('wrote', args.out)
    return out


if __name__ == '__main__':
    main()
