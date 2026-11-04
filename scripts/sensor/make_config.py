"""Write the sensor forward-model config: P_y over the training cubes, the demosaic error table, device shas.
Run after Task 5's data_loader/hfd_sensor_sim.py exists. Refuses to overwrite --out."""
import argparse
import json
import os
import time

import numpy as np

from data_loader import sensor_config as SC
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, file_sha256


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=SC.CONFIG_PATH)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--n_e2_files', type=int, default=200)
    ap.add_argument('--limit_files', type=int, default=0, help='debug: P_y over the first N training cubes')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise FileExistsError(f'{a.out} exists; refusing to overwrite')

    from data_loader.HFD_dataset import HFD_data
    from data_loader.hfd_sensor_sim import load_cube01   # lazy: Task 5 module
    from data_loader.mmse import demosaic
    from data_loader.sensor_model import DEVICES, load_sensor
    from data_loader.sensor_forward import measure

    files = HFD_data(a.data_root, train_mode='image', split='train', data_format='image', R_n=1).img_list
    if a.limit_files:
        files = files[:a.limit_files]
    R = load_sensor('device1', a.data_root)['R']
    rng = np.random.default_rng(a.seed)

    sq_sum, n_px = 0.0, 0
    e2_sum, n_e2 = np.zeros(30), 0
    for i, f in enumerate(files):
        x = load_cube01(f)
        y = x @ R
        sq_sum += float((y ** 2).sum())
        n_px += y.shape[0] * y.shape[1] * y.shape[2]
        if i < a.n_e2_files:
            ym, mask = measure(x, R, 'snapshot', 0.0, rng)
            err = (demosaic(ym, mask) - y) ** 2
            e2_sum += err.reshape(-1, err.shape[-1]).sum(0) / (err.shape[0] * err.shape[1])
            n_e2 += 1
    P_y = sq_sum / n_px

    devices = {}
    for d in DEVICES:
        s = load_sensor(d, a.data_root)
        devices[d] = dict(sha256=s['sha256'], file=s['file'], n_states_total=int(s['n_states_total']),
                          selected_states=[int(v) for v in s['selected_states']])
    cfg = SC.default_training_fields(file_sha256(a.prior))
    cfg.update(P_y=float(P_y), seed=a.seed, devices=devices, e2_demosaic=(e2_sum / max(n_e2, 1)).tolist(),
               n_files_P_y=len(files), created=time.strftime('%Y-%m-%dT%H:%M:%S'), prior_path=a.prior)
    SC.check_config(cfg)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)
    os.replace(tmp, a.out)
    print(f'P_y = {P_y:.6f}  training_sha = {SC.training_sha(cfg)}')


if __name__ == '__main__':
    main()
