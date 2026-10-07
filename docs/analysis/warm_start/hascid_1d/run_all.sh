#!/usr/bin/env bash
# End-to-end: data/prior -> condition-scale check -> sensor side check -> sweep (W workers x T threads) -> aggregate.
# CPU only. Usage: bash run_all.sh [N_PIXELS] [WORKERS] [THREADS_PER_WORKER] [CHUNK]
# The sweep is resumable: finished chunks in chunks/ are skipped. Total threads = WORKERS*THREADS (keep <= 16).
# To recompute from scratch delete data/ chunks/ chunks_addon/ first (~5.6 h + 1.2 h on the shared CPU).
# Cheap checks that do not need the full sweep:
#   python verify_chunk.py --chunk 0        (re-runs 3 configs of chunk 0; reproduced bit-exactly, ~3 min)
#   python diag_full_ddpm.py --chunk 0 --seed 0   (replays a diverging full-DDPM trajectory, ~25 min)
# (The original run executed the add-on block through run_addon.sh, because this file was edited while running.)
set -euo pipefail
N=${1:-32}; W=${2:-8}; T=${3:-2}; CH=${4:-4}
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
D="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
cd "$D"; mkdir -p logs chunks data
[ -f data/gauss_prior.npz ] || $PY -u prep_data.py > logs/prep_data.log 2>&1
[ -f data/cond_scale.json ] || NTHREADS=$T $PY -u cond_check.py > logs/cond_check.log 2>&1
[ -f data/rdevice_compare.json ] || $PY -u sensor_compare.py > logs/sensor_compare.log 2>&1
pids=()
for w in $(seq 0 $((W-1))); do
  $PY -u sweep.py --n_pixels "$N" --chunk "$CH" --threads "$T" --worker "$w" --n_workers "$W" \
      > "logs/sweep_w${w}.log" 2>&1 &
  pids+=($!); sleep 20     # stagger the 1.6 GB checkpoint loads
done
for p in "${pids[@]}"; do wait "$p"; done
# add-on: degraded linear estimate 'linreg' (s = 0.02 = 1% of the training-prior rms sensor value 1.96),
# same pixels / chunks / noise streams, no baselines
pids=()
for w in $(seq 0 $((W-1))); do
  $PY -u sweep.py --n_pixels "$N" --chunk "$CH" --threads "$T" --worker "$w" --n_workers "$W" \
      --linreg_s 0.02 --out_dir "$D/chunks_addon" > "logs/sweep_addon_w${w}.log" 2>&1 &
  pids+=($!); sleep 20
done
for p in "${pids[@]}"; do wait "$p"; done
$PY -u aggregate.py > logs/aggregate.log 2>&1
cat logs/aggregate.log
