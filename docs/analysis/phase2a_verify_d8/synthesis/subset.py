import json
s=json.load(open('/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/tests/data/val_subset_200.json'))
print({k:(len(v) if isinstance(v,list) else v) for k,v in s.items()})
dd=s.get('ddpm1000_subset')
if dd:
    imgs=set(f.rsplit('_',1)[0] for f in dd)
    print('ddpm subset n', len(dd), 'images', len(imgs), 'in gate', len(set(dd)&set(s['files'])))
