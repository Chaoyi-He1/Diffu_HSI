"""Cross-fitting split of the training cubes by SOURCE IMAGE (spec §4.1).

Source images (class/<img>, scripts.paired_stats.source_image) are grouped by class, shuffled with `seed` inside each
class and dealt alternately to half1 and half2, so no image has patches in both halves and the halves of a class
differ by at most one image. One split serves every stride. CLI (from the repo root):

    python scripts/uq/split_halves.py [--seed 0] [--out $RESIDUAL_WARMSTART_RESULTS/uq/split_halves.json]

reads the training file list exactly as the trainer does (HFDResidualData split='train' at d = 4; the file list does
not depend on d) and writes relative paths (relative to --data_root). Refuses to overwrite.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, SPLIT_JSON, git_head  # noqa: E402
from scripts.paired_stats import source_image  # noqa: E402


def split_by_source_image(files, seed=0):
    by_class = {}
    for f in files:
        key = source_image(f)
        by_class.setdefault(key.split('/')[0], {}).setdefault(key, []).append(f)
    rng = np.random.default_rng(seed)
    half1, half2, per_class = [], [], {}
    for cls in sorted(by_class):
        images = sorted(by_class[cls])
        order = rng.permutation(len(images))
        h1 = [images[i] for i in order[0::2]]
        h2 = [images[i] for i in order[1::2]]
        per_class[cls] = dict(half1=len(h1), half2=len(h2))
        for img in h1:
            half1 += by_class[cls][img]
        for img in h2:
            half2 += by_class[cls][img]
    return dict(seed=int(seed), half1=sorted(half1), half2=sorted(half2), images_per_class=per_class)


def write_split(path, split):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; the split is written once and never overwritten')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(split, f, indent=1)
    os.replace(tmp, path)


def read_split(path):
    with open(path) as f:
        return json.load(f)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=SPLIT_JSON)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--prior', default=PRIOR)
    return ap


def main():
    args = get_parser().parse_args()
    from data_loader.hfd_residual import HFDResidualData
    ds = HFDResidualData(data_path=args.data_root, split='train', sensor_down_sample_rate=4, prior_path=args.prior)
    files = [os.path.relpath(f, args.data_root) for f in ds.img_list]
    split = split_by_source_image(files, args.seed)
    split.update(n_files=len(files), data_root=args.data_root, git_head=git_head())
    write_split(args.out, split)
    print(f"wrote {args.out}: {len(split['half1'])} + {len(split['half2'])} files, images per class {split['images_per_class']}")


if __name__ == '__main__':
    main()
