"""Write the Phase 3b scenario constants (data_loader.scenarios.scenario_consts): the mean, the maximum and the
quantiles of the CLEAN Device-1 sequential readings x01 @ R of the calibration cubes (fit_b_kernel.calibration_files;
never a test cube). Shot noise reads mean_y (gamma = mean_y / sigma^2), saturation the quantiles, quantisation y_max.
The file is written once to <SENSOR_DIR>/scenario_consts.json and is what sd_eval.py --scenario_consts reads.

    python scripts/sensor/make_scenario_consts.py
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.hfd_sensor_sim import load_cube01  # noqa: E402
from data_loader.scenarios import scenario_consts  # noqa: E402
from data_loader.sensor_config import SENSOR_DIR  # noqa: E402
from data_loader.sensor_model import load_sensor  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, SUBSET, git_head  # noqa: E402

DEFAULT_OUT = os.path.join(SENSOR_DIR, 'scenario_consts.json')
DEVICE, MODE = 'device1', 'sequential'


def build_consts(files, data_root, sensor):
    """files: paths relative to data_root; sensor: load_sensor(DEVICE, data_root). -> the record (constants + provenance)."""
    clean = [load_cube01(os.path.join(data_root, f)) @ sensor['R'] for f in files]
    c = scenario_consts(clean)
    return dict(c, files=list(files), n_cubes=len(files), sensor_device=sensor['device'], mode=MODE, readings='clean',
                R_sha256=sensor['sha256'], git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'))


def write_consts(path, record):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; the scenario constants are written once (pick another --out)')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--subset', default=SUBSET)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise FileExistsError(f'{a.out} exists; the scenario constants are written once (pick another --out)')
    from scripts.sensor.fit_b_kernel import calibration_files
    rec = build_consts(calibration_files(a.subset), a.data_root, load_sensor(DEVICE, a.data_root))
    write_consts(a.out, rec)
    print(f"wrote {a.out}: {rec['n_cubes']} cubes, mean_y {rec['mean_y']:.5f}, y_max {rec['y_max']:.5f}, "
          f"quantiles {rec['quantiles']}")
    return rec


if __name__ == '__main__':
    main()
