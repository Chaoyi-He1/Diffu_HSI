"""Forward-model config for the sensor-degradation pipeline (Phase 3). The training sha over TRAIN_KEYS is what
checkpoints must match; scripts/sensor/make_config.py writes the file."""
import hashlib
import json
import os

from data_loader.warmstart_paths import RESULTS_ROOT

SENSOR_DIR = os.path.join(RESULTS_ROOT, 'sensor')
CONFIG_PATH = os.path.join(SENSOR_DIR, 'config.json')

TRAIN_KEYS = ('prior_sha256', 'axis_rule', 'normalisation', 'state_selection', 'P_y', 's_floor01', 'snr_floor_db',
              'family', 'device_train', 'tile', 'seed')
ALL_KEYS = TRAIN_KEYS + ('devices', 'e2_demosaic', 'n_files_P_y', 'created', 'prior_path')


def default_training_fields(prior_sha):
    return dict(prior_sha256=prior_sha, axis_rule='legacy', normalisation='column-minmax',
                state_selection='linspace30', s_floor01=0.01, snr_floor_db=60.0,
                family=dict(snr_low_db=20.0, snr_high_db=50.0, p_floor=0.1, p_snapshot=0.5),
                device_train='device1', tile=[5, 6], seed=0)


def check_config(cfg):
    missing = [k for k in ALL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f'sensor config is missing keys: {missing}')


def load_config(path=CONFIG_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'sensor config not found: {path} (run scripts/sensor/make_config.py)')
    with open(path) as f:
        cfg = json.load(f)
    check_config(cfg)
    return cfg


def training_sha(cfg):
    blob = json.dumps({k: cfg[k] for k in TRAIN_KEYS}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()
