#!/bin/bash
# End-to-end smoke test on 64 training files: train 2 epochs, resume to 3, quick-grid evaluation, meta refusal.
# Usage: smoke_test.sh MODE GPU            (single GPU)      ~10 min
#        smoke_test.sh MODE fsdp           (both GPUs, FSDP) ~10 min
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
MODE=${1:-residual}; TARGET=${2:-0}
if [[ "$TARGET" == "fsdp" ]]; then
  LAUNCH="$PY -m torch.distributed.run --standalone --nproc_per_node=2"; DEV=""; EVAL_GPU=0; TAG=fsdp; BS=4
else
  LAUNCH="$PY"; DEV="--device cuda:$TARGET"; EVAL_GPU=$TARGET; TAG=gpu$TARGET; BS=8
fi
RUN=$ROOT/runs/smoke_${MODE}_${TAG}_d4_b16
rm -rf "$RUN"
COMMON="main_2d_residual_fsdp.py --mode $MODE --ds 4 --base_channels 16 --batch_size $BS --num_workers 8 $DEV --limit_files 64 --save_every 1 --keep_epochs 1 2 3 --val_every 1 --save_dir $RUN"
$LAUNCH $COMMON --num_epochs 2
[[ -f "$RUN/DONE" ]] || { echo 'no DONE after 2 epochs'; exit 1; }
[[ -f "$RUN/checkpoint_epoch_2.pth" ]] || { echo 'no epoch-2 checkpoint'; exit 1; }
$LAUNCH $COMMON --num_epochs 3 --resume auto                      # resumes from epoch 2, trains epoch 3
[[ "$(cat "$RUN/DONE")" == "3" ]] || { echo 'resume did not reach epoch 3'; exit 1; }
[[ $(wc -l < "$RUN/train_log.jsonl") -eq 3 ]] || { echo 'train_log.jsonl should have 3 lines'; exit 1; }
$PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode "$MODE" --base_channels 16 --grid quick --seeds 0 --batch_size 8 --device cuda:$EVAL_GPU
JSON=$ROOT/eval/$MODE/d4/smoke_${MODE}_${TAG}_d4_b16__checkpoint_epoch_3__quick.json
[[ -f "$JSON" ]] || { echo "missing $JSON"; exit 1; }
grep -q '"meta_checked": true' "$JSON" || { echo 'meta not checked'; exit 1; }
WRONG=standard; [[ "$MODE" == "standard" ]] && WRONG=residual
if $PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode $WRONG --base_channels 16 --grid quick --seeds 0 --device cuda:$EVAL_GPU 2>/dev/null; then
  echo 'mode mismatch was NOT refused'; exit 1
fi
echo "smoke test ($MODE, $TAG) passed"
