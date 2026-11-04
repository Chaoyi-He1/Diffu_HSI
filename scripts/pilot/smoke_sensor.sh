#!/bin/bash
# CPU end-to-end smoke of Phase 3 on 16 training files (base 16, 1 epoch each): sensor config -> the four trainings
# (uncond, cond_fam, regress_fam, varhead_fam) -> sd_eval (sequential and snapshot, 30 dB) -> sd_verdict --report_only --min_common 2 ->
# toy_guidance -> Stage 0 on 4 cubes. Needs the HFD dataset and the real prior. Everything is written under its own
# temp root $T (RESIDUAL_WARMSTART_RESULTS is exported to it, so nothing touches the real results); $T is removed at
# the end on success and kept, with its path printed, on failure. ~10 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PILOT_PY:-/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python}
REAL_ROOT=${RESIDUAL_WARMSTART_RESULTS:-/data/chaoyi_he/HSI/Diffu/results/residual_warmstart}
T=$REAL_ROOT/smoke_sensor_$(date +%Y%m%d_%H%M%S)
mkdir -p "$T/sensor" "$T/runs"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "SMOKE FAILED; kept $T for inspection" >&2; fi' EXIT
cp "$REAL_ROOT/linear_prior_R1.npz" "$T/"                       # the prior (with its sigma_d table) at the redirected PRIOR path
export RESIDUAL_WARMSTART_RESULTS=$T CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
CFG=$T/config.json                                 # not $T/sensor/config.json: make_config refuses --limit_files at the (redirected) default path
need() { [[ -f "$1" ]] || { echo "missing expected output: $1"; exit 1; }; }

$PY scripts/sensor/make_config.py --out "$CFG" --limit_files 16 --n_e2_files 4
need "$CFG"

COMMON="--ds 1 --sensor_config $CFG --base_channels 16 --batch_size 4 --num_workers 2 --device cpu --num_epochs 1 --save_every 1 --val_every 1 --limit_files 16"
train() {   # mode prediction_type loss_type
  $PY main_2d_residual_fsdp.py --mode "$1" --prediction_type "$2" --loss_type "$3" $COMMON --save_dir "$T/runs/$1"
  need "$T/runs/$1/checkpoint_epoch_1.pth"
}
train uncond v l1
train cond_fam v l1
train regress_fam direct l1
train varhead_fam direct nll

CK=$T/runs
EVAL="--regress $CK/regress_fam/checkpoint_epoch_1.pth --varhead $CK/varhead_fam/checkpoint_epoch_1.pth --cond $CK/cond_fam/checkpoint_epoch_1.pth --uncond $CK/uncond/checkpoint_epoch_1.pth --K 3 --pigdm_steps 2 --limit_files 8 --subset ddpm --sensor_config $CFG --device cpu --num_workers 0"
$PY scripts/sensor/sd_eval.py --name smoke_seq_30 --sensor_device device1 --mode sequential --snr_db 30 --methods L D V C C1 B $EVAL
$PY scripts/sensor/sd_eval.py --name smoke_snap_30 --sensor_device device1 --mode snapshot --snr_db 30 --methods L L_demosaic D V C C1 B $EVAL
need "$T/sensor/eval/smoke_seq_30.json"; need "$T/sensor/eval/smoke_snap_30.json"

cat > "$T/manifest.json" <<JSON
{"name": "smoke", "records": {"device1/sequential/30": ["smoke_seq_30"], "device1/snapshot/30": ["smoke_snap_30"]}}
JSON
# --min_common 2: the records share 8 files, below the pre-registered MIN_COMMON = 45 (recorded in the report); this runs
# every clause (T1-T4, sanity) end to end on the real records
$PY scripts/sensor/sd_verdict.py --manifest "$T/manifest.json" --report_only --min_common 2
V=$T/sensor/verdict__smoke__report.json
need "$V"
$PY -c "import json,sys; v=json.load(open(sys.argv[1])); assert v['main_line'] is None and v['min_common']==2 and v['report_only'], v" "$V"

$PY scripts/sensor/toy_guidance.py --out "$T/toy.json" --steps 5 --K 8 --B 1 --hw 4 --device cpu
need "$T/toy.json"

$PY scripts/sensor/stage0_baselines.py --sensor_config "$CFG" --device cpu --limit_files 4
need "$T/sensor/stage0/summary.json"
need "$T/sensor/stage0/Lraw__device1__snapshot__20.json"

echo SMOKE OK
rm -rf "$T"
