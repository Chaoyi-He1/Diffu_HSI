import json
import sys

P = '/tmp/claude-629001453/-data-chaoyi-he-HSI-Diffu--claude-worktrees-phase1-plan/97dc7552-4c00-4553-9684-2fe175fbd44d/tasks/whszt3tpf.output'
raw = open(P).read()
obj = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
obj = obj.get('result', obj)
s = obj['synthesis']
for k in ('summary', 'confirmed', 'dose_response', 'mechanism_repeats', 'uncertain', 'adds_to_conclusion'):
    v = s[k]
    print(f'===== {k}')
    if isinstance(v, list):
        for i, x in enumerate(v, 1):
            print(f'{i}. {x}')
    else:
        print(v)
if len(sys.argv) > 1:
    for v in obj['verifiers']:
        if v['lens'] == sys.argv[1]:
            for c in v['claims']:
                print(f"- [{c['verdict']}] {c['id']}: {c['statement']}\n    {c['evidence'][:1200]}")
