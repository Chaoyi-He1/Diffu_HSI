"""The pre-registered checkpoint-selection rule (B.1, spec §4.2 / §4.5): the epoch with the lowest finite monitoring loss in
train_log.jsonl, ties -> the earlier epoch. Usage: python scripts/uq/select_epoch.py RUN_DIR"""
import json
import math
import os
import sys


def select_epoch(run_dir):
    with open(os.path.join(run_dir, 'train_log.jsonl')) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    vals = [(r['epoch'], float(r['val_loss'])) for r in recs
            if r.get('val_loss') is not None and math.isfinite(float(r['val_loss']))]
    if not vals:
        raise ValueError(f'{run_dir}: no monitoring loss in train_log.jsonl')
    epoch, val = min(vals, key=lambda t: (t[1], t[0]))
    return dict(epoch=int(epoch), val_loss=val, last_epoch=int(vals[-1][0]), last_val_loss=vals[-1][1], n_val=len(vals))


if __name__ == '__main__':
    print(json.dumps(select_epoch(sys.argv[1])))
