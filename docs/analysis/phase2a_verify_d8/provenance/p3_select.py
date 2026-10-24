import json, os, sys
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.uq.select_epoch import select_epoch
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/runs'
for run, used in [('regress_l1_sel_d8_b64', 11), ('regress_l1_sel_s43_d8_b64', 12), ('regress_l1_half1_d8_b64', 15),
                  ('regress_l1_half2_d8_b64', 13), ('varhead_d8_b64', 10), ('residual_reg_d8_b64', 100)]:
    d = os.path.join(R, run)
    recs = [json.loads(l) for l in open(os.path.join(d, 'train_log.jsonl')) if l.strip()]
    vals = [(r['epoch'], r.get('val_loss')) for r in recs if r.get('val_loss') is not None]
    s = select_epoch(d)
    # independent argmin
    ep_min = min(vals, key=lambda t: (float(t[1]), t[0]))
    srt = sorted(vals, key=lambda t: float(t[1]))[:4]
    print(f"{run}: used {used}; select_epoch -> {s['epoch']} (val {s['val_loss']:.6f}); indep argmin {ep_min}; "
          f"n_val {len(vals)} last {vals[-1]}; epochs logged {recs[0]['epoch']}..{recs[-1]['epoch']} (n={len(recs)}); best4 {srt}")
    keys = set()
    for r in recs:
        keys |= set(r.keys())
    print('   log keys:', sorted(keys))
    if run == 'residual_reg_d8_b64':
        print('   val trajectory:', [(e, round(float(v), 5)) for e, v in vals])
        print('   train loss last 5:', [(r['epoch'], round(r['loss'], 5)) for r in recs[-5:]])
    if run == 'varhead_d8_b64':
        print('   val trajectory:', [(e, round(float(v), 5)) for e, v in vals])
