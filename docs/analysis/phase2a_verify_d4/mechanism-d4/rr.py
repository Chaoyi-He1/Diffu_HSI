import json, sys
P = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/docs/analysis/phase2a_verify_d8/workflow_result.json'
obj = json.load(open(P))
obj = obj.get('result', obj)
what = sys.argv[1] if len(sys.argv) > 1 else 'list'
if what == 'list':
    print(obj.keys())
    for v in obj['verifiers']:
        print('LENS', v['lens'], len(v['claims']), v.get('scripts_written'))
elif what == 'synthesis':
    s = obj['synthesis']
    for k, v in s.items():
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
            print(f"- [{c['verdict']}] {c['id']}: {c['statement']}\n    {c['evidence']}")
        print('DISCREPANCIES:')
        for d in v['discrepancies']:
            print('  *', d)
        print('SCRIPTS:', v.get('scripts_written'))
