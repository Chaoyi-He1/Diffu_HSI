#!/bin/bash
# GPU smoke test of mode regress on 64 training files: train 2 epochs, resume to 3, evaluate the direct grid
# (meta-checked, loss recorded), and confirm that evaluating the regressor as B is refused. Usage: smoke_regress.sh GPU
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python
ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
GPU=${1:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
NAME=smoke_regress_gpu${GPU}_d4_b16
RUN=$ROOT/runs/$NAME
rm -rf "$RUN"
COMMON="main_2d_residual_fsdp.py --mode regress --prediction_type direct --loss_type l2 --ds 4 --base_channels 16 --batch_size 8 --num_workers 8 --device cuda:$GPU --limit_files 64 --save_every 1 --keep_epochs 1 2 3 --val_every 1 --save_dir $RUN"
$PY $COMMON --num_epochs 2
[[ "$(cat "$RUN/DONE")" == "2" ]] || { echo 'no DONE = 2 after 2 epochs'; exit 1; }
$PY $COMMON --num_epochs 3 --resume auto
[[ "$(cat "$RUN/DONE")" == "3" ]] || { echo 'resume did not reach epoch 3'; exit 1; }
$PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode regress --prediction_type direct \
    --grid direct --base_channels 16 --seeds 0 1 --batch_size 8 --device "cuda:$GPU"
JSON=$ROOT/eval/regress/d4/${NAME}__checkpoint_epoch_3__direct.json
[[ -f "$JSON" ]] || { echo "missing $JSON"; exit 1; }
grep -q '"meta_checked": true' "$JSON" || { echo 'meta not checked'; exit 1; }
grep -q '"loss_type": "l2"' "$JSON" || { echo 'loss_type not recorded'; exit 1; }
if ERR=$($PY scripts/eval_warmstart.py --d 4 --ckpt "$RUN/checkpoint_epoch_3.pth" --mode residual --grid full \
         --base_channels 16 --seeds 0 --device "cuda:$GPU" 2>&1); then
  echo 'evaluating the regressor as B was NOT refused'; exit 1
fi
grep -q 'does not match this evaluation' <<< "$ERR" || { echo "refused for another reason: $ERR"; exit 1; }
rm -rf "$RUN" "$JSON"
echo "regress smoke test (GPU $GPU) passed"
