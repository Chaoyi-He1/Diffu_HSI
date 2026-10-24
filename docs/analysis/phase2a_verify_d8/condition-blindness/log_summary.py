import json, sys
for run in sys.argv[1:]:
    p = f'/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/runs/{run}/train_log.jsonl'
    rows = [json.loads(l) for l in open(p)]
    print(run, len(rows), 'keys', sorted(set().union(*[r.keys() for r in rows])))
    for r in rows:
        extra = {k: v for k, v in r.items() if k.startswith('val') or 'mon' in k}
        print(r.get('epoch'), round(r.get('loss', float('nan')), 5), round(r.get('lr', 0), 7),
              {k: (round(v, 5) if isinstance(v, float) else v) for k, v in extra.items()})
