#!/bin/bash
# Launch one pilot run in the background on one GPU. Usage: launch_run.sh residual|standard|standard_xhat GPU [extra args]
# Refuses (exit 1; nothing started, the launch not counted) when the GPU is busy, another launch_run.sh holds the run's
# .launch.lock, the run has a DONE file (delete it to train further), or its pid file names a live process.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PILOT_PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python}     # tests point PILOT_PY at a stub
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
MODE=$1; GPU=$2; shift 2
RUN=$ROOT/runs/${MODE}_d4_b64
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU")
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$GPU" | sed '/^$/d' | wc -l)
if [[ "$FREE" -lt 3000 || "$PROCS" -gt 0 ]]; then echo "GPU $GPU busy (free ${FREE} MiB, ${PROCS} procs); not launching"; exit 1; fi
mkdir -p "$RUN"
exec 9>"$RUN/.launch.lock"                  # one launcher per run at a time; released when this script exits (the trainer gets 9>&-)
flock -n 9 || { echo "another launch_run.sh holds $RUN/.launch.lock; not launching"; exit 1; }
if [[ -f "$RUN/DONE" ]]; then echo "$RUN/DONE exists; not launching (delete DONE to train further)"; exit 1; fi
if [[ -f "$RUN/pid" ]] && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
  echo "$MODE is already running (pid $(cat "$RUN/pid")); not launching"; exit 1
fi
N=$(cat "$RUN/launches" 2>/dev/null || echo 0)
[[ "$N" =~ ^[0-9]+$ ]] || N=0
echo $(( 10#$N + 1 )) > "$RUN/launches"     # EVERY launch counts (even before the first checkpoint); the driver stops after 1 + MAX_RELAUNCHES
RESUME=""
if compgen -G "$RUN/checkpoint_epoch_*.pth" > /dev/null; then
  RESUME="--resume auto"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup $PY main_2d_residual_fsdp.py --mode "$MODE" --ds 4 --base_channels 64 --batch_size 8 --num_workers 8 --device "cuda:$GPU" \
    --num_epochs 100 --max_hours 24 --save_every 10 --keep_epochs 25 50 100 --val_every 5 --save_dir "$RUN" $RESUME "$@" \
    >> "$RUN/run.log" 2>&1 9>&- &
echo $! > "$RUN/pid"
echo "launched $MODE on GPU $GPU (pid $(cat "$RUN/pid"), resume='${RESUME}') -> $RUN/run.log"
