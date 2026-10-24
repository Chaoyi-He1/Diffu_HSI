#!/bin/bash
# Launch one pilot run in the background on one GPU.
# Usage: [RUN_NAME=<dir name>] launch_run.sh residual|standard|standard_xhat|regress|residual_reg|varhead GPU [extra args]   (regress needs --prediction_type direct --loss_type l1|l2)
# Refuses (exit 1; nothing started, the launch not counted) when a non-empty RUN_NAME is not <MODE>_[A-Za-z0-9_]+ (no
# '/' or '..'; an exported RUN_NAME cannot redirect another mode's launch) or belongs to a longer mode name that starts
# with <MODE>_ (residual_reg_* for residual, standard_xhat_* for standard), when a regress launch has no --loss_type l1|l2
# (the trainer's default is l1) or one that contradicts an _l1_ / _l2_ in RUN_NAME, when a regress RUN_NAME names a seed
# (_s<N>_, or _s<N> at its end) that --seed does not pass, or a regress --seed other than the trainer's default 42 is
# given under a RUN_NAME that names none (the last --seed N / --seed=N wins, as in argparse), when a regress RUN_NAME
# with `_half<N>` does not pass `--train_half N` (and `--train_half N` needs the token), when a varhead launch lacks
# `--loss_type nll`, when the GPU is busy,
# another launch_run.sh holds the run's .launch.lock, the run has a DONE file (delete it to train further), or its pid
# file names a live process.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PILOT_PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python}     # tests point PILOT_PY at a stub
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
MODE=$1; GPU=$2; shift 2
NAME_RE="^${MODE}_[A-Za-z0-9_]+\$"
if [[ -n "${RUN_NAME:-}" && ! "$RUN_NAME" =~ $NAME_RE ]]; then
  echo "RUN_NAME='$RUN_NAME' is not ${MODE}_[A-Za-z0-9_]+; not launching"; exit 1
fi
for M in residual standard standard_xhat regress residual_reg varhead; do      # model/residual_wrapper.py MODES
  if [[ "$M" != "$MODE" && "$M" == "${MODE}_"* && "${RUN_NAME:-}" == "${M}_"* ]]; then   # e.g. residual_reg_* passes ^residual_
    echo "RUN_NAME='$RUN_NAME' belongs to mode $M, not $MODE; not launching"; exit 1
  fi
done
LOSS=""; SEED=""; HALF=""; PREV=""
for A in "$@"; do                                         # the last --loss_type / --seed / --train_half wins, as in argparse
  [[ "$PREV" == --loss_type ]] && LOSS=$A
  [[ "$A" == --loss_type=* ]] && LOSS=${A#--loss_type=}
  [[ "$PREV" == --seed ]] && SEED=$A
  [[ "$A" == --seed=* ]] && SEED=${A#--seed=}
  [[ "$PREV" == --train_half ]] && HALF=$A
  [[ "$A" == --train_half=* ]] && HALF=${A#--train_half=}
  PREV=$A
done
if [[ "$MODE" == regress ]]; then
  if [[ "$LOSS" != l1 && "$LOSS" != l2 ]]; then
    echo "regress needs --loss_type l1|l2 in the extra args (got '${LOSS}'; RUN_NAME='${RUN_NAME:-}'); not launching"; exit 1
  fi
  for L in l1 l2; do
    if [[ "${RUN_NAME:-}" == *_${L}_* && "$LOSS" != "$L" ]]; then
      echo "RUN_NAME='$RUN_NAME' names $L but --loss_type is $LOSS; not launching"; exit 1
    fi
  done
  if [[ -n "$SEED" && ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "--seed '$SEED' is not a non-negative integer; not launching"; exit 1
  fi
  SEED_RE='_s([0-9]+)(_|$)'
  if [[ "${RUN_NAME:-}" =~ $SEED_RE ]]; then              # a seed-N run (e.g. regress_l1_s43_d4_b64) must pass --seed N
    NAME_SEED=${BASH_REMATCH[1]}
    if [[ -z "$SEED" ]] || (( 10#$SEED != 10#$NAME_SEED )); then
      echo "RUN_NAME='$RUN_NAME' names seed $NAME_SEED but --seed is '$SEED'; not launching"; exit 1
    fi
  elif [[ -n "$SEED" ]] && (( 10#$SEED != 42 )); then     # names without _s<N>_ are the trainer's default seed 42
    echo "--seed $SEED but RUN_NAME='${RUN_NAME:-}' names no _s<N>_ (only seed-42 runs omit it); not launching"; exit 1
  fi
  HALF_RE='_half([12])(_|$)'
  if [[ "${RUN_NAME:-}" =~ $HALF_RE ]]; then              # a half-data regressor (Phase 2a) must pass --train_half N
    NAME_HALF=${BASH_REMATCH[1]}
    if [[ "$HALF" != "$NAME_HALF" ]]; then
      echo "RUN_NAME='$RUN_NAME' names half $NAME_HALF but --train_half is '$HALF'; not launching"; exit 1
    fi
  elif [[ -n "$HALF" && "$HALF" != 0 ]]; then
    echo "--train_half $HALF but RUN_NAME='${RUN_NAME:-}' names no _half<N>_ (half regressors carry it); not launching"; exit 1
  fi
fi
if [[ "$MODE" == varhead && "$LOSS" != nll ]]; then       # the varhead's loss is the Gaussian NLL; the trainer's default is l1
  echo "varhead needs --loss_type nll in the extra args (got '${LOSS}'); not launching"; exit 1
fi
RUN=$ROOT/runs/${RUN_NAME:-${MODE}_d4_b64}     # RUN_NAME overrides the run directory (e.g. regress_l2_d4_b64)
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
