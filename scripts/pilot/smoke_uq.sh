#!/bin/bash
# CPU end-to-end smoke of Phase 2a on 16 training files (base 8, 1 epoch each): split -> half regressors -> full
# regressors (seeds 42, 43) -> centre cache -> residual_reg -> varhead -> uq_eval (K = 2) -> uq_verdict (--no_centre_check).
# Needs the HFD dataset and the real prior; writes only under $SMOKE_UQ_TMP. ~10 min. SMOKE_UQ_TMP names the smoke's own
# directory (default /tmp/smoke_uq_<pid>); it is deleted and recreated, and an existing directory without the smoke's
# marker (.smoke_uq) is refused.
# K = 2 < min_k_for(0.9) = 19: uq_eval's coverage warnings and uq_verdict's report-only WARNING are expected.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PILOT_PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python}
REAL_ROOT=${REAL_RESULTS_ROOT:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
T=${SMOKE_UQ_TMP:-/tmp/smoke_uq_$$}
if [[ -e "$T" && ! -f "$T/.smoke_uq" ]]; then echo "$T exists and was not made by smoke_uq.sh; refusing" >&2; exit 1; fi
rm -rf "$T"; mkdir -p "$T/root/runs" "$T/root/uq"; touch "$T/.smoke_uq"
cp "$REAL_ROOT/linear_prior_R1.npz" "$T/root/"                    # the prior with its sigma_d table
export RESIDUAL_WARMSTART_RESULTS=$T/root CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
R=$T/root/runs; SPLIT=$T/root/uq/split_halves.json; CACHE=$T/root/uq/d4/xr_cache
COMMON="--device cpu --base_channels 8 --batch_size 4 --num_workers 2 --num_epochs 1 --save_every 1 --val_every 1 --keep_last 2 --no_bf16 --ds 4"
REG="--mode regress --prediction_type direct --loss_type l1 --split_json $SPLIT"

$PY scripts/uq/split_halves.py --out "$SPLIT"
$PY main_2d_residual_fsdp.py $REG $COMMON --train_half 1 --limit_files 8 --save_dir "$R/regress_l1_half1_d4_b8"
$PY main_2d_residual_fsdp.py $REG $COMMON --train_half 2 --limit_files 8 --save_dir "$R/regress_l1_half2_d4_b8"
$PY main_2d_residual_fsdp.py $REG $COMMON --limit_files 16 --save_dir "$R/regress_l1_full_d4_b8"
# uq_eval refuses half regressors as ensemble members (train_half must be 0): a second full-data seed is the 2nd member
$PY main_2d_residual_fsdp.py $REG $COMMON --limit_files 16 --seed 43 --save_dir "$R/regress_l1_full_s43_d4_b8"
for run in regress_l1_half1_d4_b8 regress_l1_half2_d4_b8 regress_l1_full_d4_b8 regress_l1_full_s43_d4_b8; do
  [[ "$(cat "$R/$run/DONE")" == "1" ]] || { echo "$run did not finish"; exit 1; }
done
$PY scripts/uq/build_xr_cache.py --d 4 --half1 "$R/regress_l1_half1_d4_b8/checkpoint_epoch_1.pth" \
    --half2 "$R/regress_l1_half2_d4_b8/checkpoint_epoch_1.pth" --full "$R/regress_l1_full_d4_b8/checkpoint_epoch_1.pth" \
    --out "$CACHE" --split_json "$SPLIT" --device cpu --batch_size 4 --num_workers 2 --limit_files 16
[[ -f "$CACHE/meta.json" ]] || { echo 'no cache meta'; exit 1; }
$PY main_2d_residual_fsdp.py --mode residual_reg $COMMON --xr_cache "$CACHE" --limit_files 16 --save_dir "$R/residual_reg_d4_b8"
$PY main_2d_residual_fsdp.py --mode varhead --prediction_type direct --loss_type nll $COMMON --xr_cache "$CACHE" --limit_files 16 --save_dir "$R/varhead_d4_b8"
for run in residual_reg_d4_b8 varhead_d4_b8; do
  [[ "$(cat "$R/$run/DONE")" == "1" ]] || { echo "$run did not finish"; exit 1; }
done
# json.dumps, not the dict's repr: the repr quotes keys with ' and would never match "sigma_r"
grep -q '"sigma_r"' <<< "$($PY -c "import json, torch; print(json.dumps(torch.load('$R/residual_reg_d4_b8/checkpoint_epoch_1.pth', map_location='cpu', weights_only=False)['meta'], default=str))")" \
    || { echo 'residual_reg meta lacks sigma_r'; exit 1; }
$PY scripts/uq_eval.py --d 4 --name smoke --centre "$R/regress_l1_full_d4_b8/checkpoint_epoch_1.pth" \
    --diffusion "$R/residual_reg_d4_b8/checkpoint_epoch_1.pth" --varhead "$R/varhead_d4_b8/checkpoint_epoch_1.pth" \
    --ensemble "$R/regress_l1_full_d4_b8/checkpoint_epoch_1.pth" "$R/regress_l1_full_s43_d4_b8/checkpoint_epoch_1.pth" \
    --xr_cache "$CACHE" --K 2 --limit_files 4 --yswap_n 4 --device cpu --batch_size 2 --num_workers 0 --store_cubes 2
REC=$T/root/uq/d4/eval/smoke.json
[[ -f "$REC" && -f "$T/root/uq/d4/eval/smoke__cubes.npz" ]] || { echo 'no record'; exit 1; }
$PY scripts/uq_verdict.py "$REC" --no_centre_check --report_only
grep -q '"S2"' "$T/root/uq/d4/verdict__smoke__nocentre__report.json" || { echo 'verdict lacks S2'; exit 1; }
if ERR=$($PY scripts/uq_verdict.py "$REC" --report_only 2>&1); then echo 'verdict accepted a non-pre-registered centre'; exit 1; fi
grep -q 'is not the pre-registered selected regressor' <<< "$ERR" || { echo "verdict refused for another reason: $ERR"; exit 1; }
echo "uq smoke passed ($T)"
