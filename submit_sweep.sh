#!/bin/bash
# Submit one SLURM job per downsample rate for the sensor-SR sweep.
# Each job runs independently (no shared state, no warm-start) and writes
# to its own results/2d_sensor_sr/ds${ds}_area/... folder.
set -euo pipefail

cd "$(dirname "$0")"

for ds in 2 3 4 6; do
    sbatch \
        --job-name="ssr_ds${ds}" \
        --export=ALL,DS_SR=${ds} \
        job_sensor_sr_sweep.slurm
done
