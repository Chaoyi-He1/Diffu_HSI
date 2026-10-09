"""Fixed, stratified subset of the loader's validation split for diffusion evaluation (spec §7).
24 cubes from P022, 152 from P023, 24 from P024; the first 50 (stratified 6/38/6) are the DDPM-1000 subset.
Run from the repo root:  python scripts/make_val_subset.py
"""
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.HFD_dataset import HFD_data  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
QUOTA = {'P022': 24, 'P023': 152, 'P024': 24}
DDPM_QUOTA = {'P022': 6, 'P023': 38, 'P024': 6}


def main(seed=0):
    ds = HFD_data(data_path=DATA_ROOT, train_mode='image', eval_ratio=0.1, split='test',
                  data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    by_cls = {}
    for f in ds.img_list:
        by_cls.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    rng = random.Random(seed)
    files, ddpm = [], []
    for c, n in QUOTA.items():
        pick = rng.sample(sorted(by_cls[c]), n)
        files += pick
        ddpm += pick[:DDPM_QUOTA[c]]
    rel = lambda f: os.path.relpath(f, DATA_ROOT)
    out = dict(seed=seed, files=[rel(f) for f in files], classes=QUOTA, ddpm1000_subset=[rel(f) for f in ddpm])
    path = os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path, len(files), 'files,', len(ddpm), 'for DDPM-1000')


if __name__ == '__main__':
    main()
