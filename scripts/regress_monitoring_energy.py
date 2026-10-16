"""Residual energy the one-step regressors remove on the MONITORING (validation) files — never the 200 gate cubes.

Phase 1a-prime, Task 8 review (the overfitting finding). For every kept checkpoint of each run
(RESULTS_ROOT/runs/<run>/checkpoint_epoch_*.pth, loaded read-only on the CPU, meta-checked) it computes, per monitoring
file, the MSE and the MAE of the regressor's single call against r = (x - x0_hat) / sigma_d, in sigma_d units on the
loader's [-1, 1] scale. That is exactly the quantity the training script's validate() averages: RegressionTrainer.get_loss
takes the per-sample mean of |r_hat - r| (l1) or (r_hat - r)^2 (l2), then the mean over samples. It also computes the
zero prediction r_hat = 0, i.e. the baseline x0_hat itself (MSE = mean r^2, MAE = mean |r|). Everything is reported per
class (P022/P023/P024) and pooled (mean over the files), with the per-file arrays. 1 - MSE / zero-prediction MSE is the
fraction of the residual energy a checkpoint removes on the monitoring files.

The monitoring set is built by the training script's own main_2d_residual_fsdp.make_datasets from each run's args.json:
the validation split with the 200 gate cubes (tests/data/val_subset_200.json) removed, the same prior and d. It is the
set behind train_log.jsonl's val_loss, which each checkpoint's own loss reproduces up to bf16 rounding. This script
never loads a gate cube: it refuses to run if any of them is in the set.

It writes only under RESULTS_ROOT/diagnostics, refuses an existing --out, and writes through a temporary file.

The record RESULTS_ROOT/diagnostics/regress_monitoring_energy.json (2026-10-16) was produced by the scratch original
of this script at commit 20dc1e9 with
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 THREADS=8 nice -n 5 python monitoring_energy.py
(8 torch threads, 4 loader workers, 1717 s). The equivalent command with this script is
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 nice -n 5 \
      python scripts/regress_monitoring_energy.py --threads 8
and it refuses to run while that record exists.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import time
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import RESULTS_ROOT, SUBSET, file_sha256, git_head  # noqa: E402

DIAG_DIR = os.path.join(RESULTS_ROOT, 'diagnostics')
DEFAULT_OUT = os.path.join(DIAG_DIR, 'regress_monitoring_energy.json')
RUNS = ('regress_l1_d4_b64', 'regress_l2_d4_b64')


def under(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def check_out(path):
    """The only place this script writes: a new file under RESULTS_ROOT/diagnostics."""
    if not under(path, DIAG_DIR):
        raise SystemExit(f'--out {path} is not under {DIAG_DIR}; this script writes only there')
    if os.path.exists(path):
        raise SystemExit(f'{path} exists; this script never overwrites it (choose another --out)')


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--runs', nargs='+', default=list(RUNS), help='run directories under RESULTS_ROOT/runs')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--threads', type=int, default=8, help='torch CPU threads')
    ap.add_argument('--workers', type=int, default=4, help='DataLoader workers (one BLAS thread each)')
    return ap


def main(argv=None):
    args = get_parser().parse_args(argv)
    check_out(args.out)                                        # before any work
    head = git_head()                                          # the HEAD at launch
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data_loader.hfd_residual import limit_worker_threads, residual_collate_fn
    from main_2d_residual_fsdp import make_datasets
    from model.residual_wrapper import build_residual_model, regress_forward, target_for

    torch.set_num_threads(args.threads)
    t_start = time.time()
    run_args = {run: json.load(open(os.path.join(RESULTS_ROOT, 'runs', run, 'args.json'))) for run in args.runs}
    keys = ('data_root', 'd', 'prior', 'limit_files')
    a0 = run_args[args.runs[0]]
    if any(any(a[k] != a0[k] for k in keys) for a in run_args.values()):
        raise SystemExit(f'the runs differ in {keys}: their monitoring sets are not the same')
    _, val = make_datasets(SimpleNamespace(data_root=a0['data_root'], d=a0['d'], prior=a0['prior'], limit_files=a0['limit_files']))
    gate = {os.path.normpath(os.path.join(a0['data_root'], f)) for f in json.load(open(SUBSET))['files']}
    if any(os.path.normpath(f) in gate for f in val.img_list):
        raise SystemExit('the monitoring set contains a gate cube; refusing')
    files = [os.path.relpath(f, a0['data_root']) for f in val.img_list]
    classes = [re.search(r'(P\d{3})', f).group(1) for f in files]
    sd = val.sigma_d
    prior_sha = file_sha256(a0['prior'])
    print(f'monitoring files {len(files)}, sigma_d {sd:.6f}, classes', {c: classes.count(c) for c in sorted(set(classes))}, flush=True)
    loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=args.workers, collate_fn=residual_collate_fn,
                        worker_init_fn=limit_worker_threads)
    batches = list(loader)
    assert sum(b[0].shape[0] for b in batches) == len(files)
    print(f'loaded in {time.time() - t_start:.0f} s', flush=True)

    zero_mse, zero_mae = [], []
    for x, y, xh in batches:
        r = target_for('regress', x, xh, sd).double()
        zero_mse += r.pow(2).reshape(len(r), -1).mean(1).tolist()
        zero_mae += r.abs().reshape(len(r), -1).mean(1).tolist()
    cls = np.array(classes)

    def summ(v):
        v = np.asarray(v, float)
        return dict(pooled=float(v.mean()), **{c: float(v[cls == c].mean()) for c in sorted(set(classes))})

    out = dict(files=files, classes=classes, n={c: int((cls == c).sum()) for c in sorted(set(classes))},
               sigma_d=sd, zero=dict(mse=summ(zero_mse), mae=summ(zero_mae), per_file=dict(mse=zero_mse, mae=zero_mae)),
               checkpoints={})
    for run in args.runs:
        log = {e['epoch']: e for e in (json.loads(l) for l in open(os.path.join(RESULTS_ROOT, 'runs', run, 'train_log.jsonl')))}
        cks = sorted(glob.glob(os.path.join(RESULTS_ROOT, 'runs', run, 'checkpoint_epoch_*.pth')),
                     key=lambda p: int(re.search(r'epoch_(\d+)', p).group(1)))
        if not cks:
            raise SystemExit(f'{run}: no checkpoint_epoch_*.pth')
        for p in cks:
            ep = int(re.search(r'epoch_(\d+)', p).group(1))
            t0 = time.time()
            ck = torch.load(p, map_location='cpu', weights_only=False)
            meta = ck.get('meta', {})
            lt = meta.get('args', {}).get('loss_type')
            if meta.get('mode') != 'regress' or meta.get('prediction_type') != 'direct' or lt != run_args[run]['loss_type']:
                raise SystemExit(f'{p}: meta mode={meta.get("mode")!r} prediction_type={meta.get("prediction_type")!r} '
                                 f'loss_type={lt!r} is not this regress run')
            if abs(meta.get('sigma_d') - sd) > 1e-12 or meta.get('prior_sha256') != prior_sha:
                raise SystemExit(f'{p}: checkpoint sigma_d / prior differ from the monitoring set\'s')
            model = build_residual_model('regress', run_args[run]['base_channels'])
            model.load_state_dict(ck['model_state_dict'], strict=True)
            model.eval()
            mse, mae = [], []
            with torch.no_grad():
                for x, y, xh in batches:
                    e = (regress_forward(model, {'sensor': y, 'x0_hat': xh}).float() - target_for('regress', x, xh, sd).float()).double()
                    mse += e.pow(2).reshape(len(e), -1).mean(1).tolist()
                    mae += e.abs().reshape(len(e), -1).mean(1).tolist()
            logged = log.get(ep, {}).get('val_loss')
            rec = dict(run=run, epoch=ep, loss_type=lt, checkpoint=p, checkpoint_sha256=file_sha256(p),
                       logged_val_loss=logged, mse=summ(mse), mae=summ(mae), per_file=dict(mse=mse, mae=mae))
            own = rec['mae' if lt == 'l1' else 'mse']['pooled']
            print(f"{run} epoch {ep}: MSE {rec['mse']['pooled']:.4f} MAE {rec['mae']['pooled']:.4f} (logged val {logged}; "
                  f"own-loss here {own:.4f}) removed {100 * (1 - rec['mse']['pooled'] / out['zero']['mse']['pooled']):.1f} % "
                  f"[{time.time() - t0:.0f} s]", flush=True)
            out['checkpoints'][f'{run}@{ep}'] = rec
            del model, ck
    out['provenance'] = dict(script=os.path.abspath(__file__), git_head=head, torch=torch.__version__, threads=args.threads,
                             device='cpu', prior=a0['prior'], prior_sha256=prior_sha, data_root=a0['data_root'],
                             d=a0['d'], built_by='main_2d_residual_fsdp.make_datasets (gate cubes removed)',
                             seconds=time.time() - t_start, finished=dt.datetime.now().isoformat(timespec='seconds'))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f)
    if os.path.exists(args.out):
        os.remove(tmp)
        raise SystemExit(f'{args.out} appeared while running; not overwritten')
    os.replace(tmp, args.out)
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
