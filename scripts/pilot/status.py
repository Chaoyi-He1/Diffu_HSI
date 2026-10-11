"""One line per run: last epoch, loss, val, lr, elapsed, ETA, alive/DONE, evaluations present."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.pilot.pilot_driver import ROOT, run_state  # noqa: E402

for run in sorted(glob.glob(os.path.join(ROOT, 'runs', '*_d4_b*'))):
    st = run_state(run, os.path.join(ROOT, 'eval'))
    recs = []
    log = os.path.join(run, 'train_log.jsonl')
    if os.path.exists(log):
        for line in open(log):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    name = os.path.basename(run)
    if not recs:
        print(f'{name:<24} no epochs yet   alive={st["alive"]} DONE={st["done"]}'); continue
    r = recs[-1]
    n = json.load(open(os.path.join(run, 'args.json')))['num_epochs'] if os.path.exists(os.path.join(run, 'args.json')) else 100
    print(f"{name:<24} epoch {r['epoch']:3d}/{n} loss {r['loss']:.4f} val {r.get('val_loss', float('nan')):.4f} lr {r['lr']:.1e} "
          f"{r['sec']:.0f}s/ep (data {100 * r['data_sec'] / max(r['sec'], 1e-9):.0f}%) elapsed {r['elapsed_hours']:.1f}h "
          f"eta {(n - r['epoch']) * r['sec'] / 3600:.1f}h skipped {r['skipped_nonfinite']} alive={st['alive']} DONE={st['done']} "
          f"quick={st['quick_evals']} full={st['full_evals']}")
