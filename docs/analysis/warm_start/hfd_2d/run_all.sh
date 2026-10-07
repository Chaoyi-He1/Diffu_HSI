#!/bin/bash
# End-to-end: warm-start / truncated sampling sweep on the HFD 64-band checkpoint (epoch 101).
# Runtime on the shared A4500 (GPU 0, ~25 ms per item-NFE when contended):
#   primary ~2 h, primary clip variants ~25 min, secondary (reduced) ~25 min.
# The sweep resumes from records_*.jsonl; delete those files for a fresh run.
set -e
D="$(cd "$(dirname "$0")" && pwd)"
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
export PYTHONDONTWRITEBYTECODE=1
cd "$D"
$PY check_load.py ${WS_OLDCODE:-/tmp/diffu_oldcode}/988a338 | tee check_load.log   # strict load
$PY check_sdpa.py | tee check_sdpa.log            # fused attention == original attention (CPU fp32)
$PY fit_prior.py | tee fit_prior.log              # test sets, Gaussian prior, choice of s  (CPU)
$PY diag_denoise.py | tee diag_denoise.log        # one-step denoising error vs t, with true / zero / shuffled y (GPU 0)
$PY diag_samples.py | tee diag_samples.log        # what pure-noise samples look like; effect of swapping y (CPU)
$PY run_sweep.py --set primary --seeds 0,1 --batch 8 2>&1 | tee -a sweep_primary.log
$PY run_sweep.py --set primary --seeds 0,1 --batch 8 --clip-x0 2>&1 | tee -a sweep_primary_clip.log   # x0-clipped sampler variants
$PY run_sweep.py --set secondary --seeds 0,1 --batch 8 --reduced 2>&1 | tee -a sweep_secondary.log
$PY aggregate.py | tee table.txt                  # -> results.json
