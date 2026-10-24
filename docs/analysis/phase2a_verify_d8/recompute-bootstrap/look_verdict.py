import json, os, collections
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8'
E = os.path.join(R, 'ev' + 'al')
v = json.load(open(os.path.join(R, 'verdict__d8_main.json')))
print(json.dumps(v, indent=1)[:4000])
sub = json.load(open('/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/tests/data/val_subset_200.json'))
print(sub.keys())
print({k: (len(x) if isinstance(x, (list, dict)) else x) for k, x in sub.items()})
print(sub.get('classes'))
rec = json.load(open(os.path.join(E, 'd8_main.json')))
print('record files == subset files (as list):', rec['files'] == sub['files'], ' as set:', set(rec['files']) == set(sub['files']))
ys = rec['yswap']
print('yswap files', ys['files'])
print('yswap in gate', all(f in set(rec['files']) for f in ys['files']))
key = lambda f: os.path.basename(os.path.dirname(f)) + '/' + os.path.basename(f).rsplit('_', 1)[0]
print('yswap clusters', len(set(map(key, ys['files']))))
print('gate clusters', len(set(map(key, rec['files']))))
print(collections.Counter(rec['classes']))
print(collections.Counter(collections.Counter(map(key, rec['files'])).values()))
print('stored', rec['stored_files'])
print(json.dumps(rec['provenance'], indent=1)[1500:6000])
