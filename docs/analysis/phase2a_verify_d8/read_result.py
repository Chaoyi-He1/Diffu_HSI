import json
import sys

P = '/tmp/claude-629001453/-data-chaoyi-he-HSI-Diffu--claude-worktrees-phase1-plan/97dc7552-4c00-4553-9684-2fe175fbd44d/tasks/w0y5vy0dt.output'
raw = open(P).read()
start = raw.find('{')
obj = json.loads(raw[start:raw.rfind('}') + 1])
obj = obj.get('result', obj)
what = sys.argv[1] if len(sys.argv) > 1 else 'synthesis'
if what == 'synthesis':
    s = obj['synthesis']
    for k in ('summary', 'confirmed', 'discrepancies_resolved', 'mechanisms', 'uncertain', 'fix_round_recommendation', 'next_experiment'):
        v = s[k]
        print(f'===== {k}')
        if isinstance(v, list):
            for i, x in enumerate(v, 1):
                print(f'{i}. {x}')
        else:
            print(v)
else:
    for v in obj['verifiers']:
        if v['lens'] != what:
            continue
        print('SUMMARY:', v['summary'])
        for c in v['claims']:
            print(f"- [{c['verdict']}] {c['id']}: {c['statement']}\n    {c['evidence'][:900]}")
        print('DISCREPANCIES:')
        for d in v['discrepancies']:
            print('  *', d[:700])
        print('SCRIPTS:', v.get('scripts_written'))
