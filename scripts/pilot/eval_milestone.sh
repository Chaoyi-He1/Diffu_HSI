#!/bin/bash
# Evaluate one pilot checkpoint. Usage: eval_milestone.sh MODE EPOCH quick|full GPU   (full requires an idle GPU)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
MODE=$1; EPOCH=$2; GRID=$3; GPU=$4
RUN=$ROOT/runs/${MODE}_d4_b64
CK=$RUN/checkpoint_epoch_${EPOCH}.pth
[[ -f "$CK" ]] || { echo "missing $CK"; exit 1; }
LOG=$RUN/eval_${GRID}_epoch_${EPOCH}.log
if [[ "$GRID" == "full" ]]; then
  PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$GPU" | sed '/^$/d' | wc -l)
  [[ "$PROCS" -eq 0 ]] || { echo "GPU $GPU busy; the full-grid evaluation needs an idle GPU"; exit 1; }
  nvidia-smi --id="$GPU" >> "$LOG"
fi
$PY scripts/eval_warmstart.py --d 4 --ckpt "$CK" --mode "$MODE" --base_channels 64 --grid "$GRID" --seeds 0 1 --batch_size 8 --device "cuda:$GPU" 2>&1 | tee -a "$LOG"
