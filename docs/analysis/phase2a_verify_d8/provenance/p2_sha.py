import hashlib, json, os, sys
import torch
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
prov = rec['provenance']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


items = [('centre', prov['centre']), ('diffusion', prov['diffusion']), ('varhead', prov['varhead']),
         ('ens0', prov['ensemble'][0]), ('ens1', prov['ensemble'][1]), ('half1', prov['halves'][0]),
         ('half2', prov['halves'][1])]
out = {}
for name, p in items:
    s = sha(p['path'])
    ck = torch.load(p['path'], map_location='cpu', weights_only=False)
    meta = ck.get('meta') or {}
    m = {k: v for k, v in meta.items() if k not in ('args',)}
    args = meta.get('args') or {}
    print(f"== {name}: {p['path']}")
    print(f"   sha disk {s}\n   sha rec  {p['sha256']}  match={s == p['sha256']}")
    print('   ckpt top keys:', [k for k in ck.keys()])
    print('   ckpt epoch field:', ck.get('epoch'))
    for k, v in m.items():
        if isinstance(v, (dict, list)) and len(json.dumps(v, default=str)) > 400:
            print(f'   meta.{k}: <{type(v).__name__} len {len(v)}>')
        else:
            print(f'   meta.{k}: {v}')
    print('   args.train_half', args.get('train_half'), 'args.seed', args.get('seed'), 'args.loss_type', args.get('loss_type'),
          'args.prediction_type', args.get('prediction_type'), 'args.xr_cache', args.get('xr_cache'), 'args.split_json', args.get('split_json'))
    out[name] = dict(sha_disk=s, sha_rec=p['sha256'], meta={k: (v if not isinstance(v, (dict, list)) else v) for k, v in m.items()})
    del ck

print('== cache meta sha', sha(R + '/uq/d8/xr_cache/meta.json'), 'rec', prov['xr_cache']['sha256'])
print('== prior sha', sha(prov['prior']['path']), 'rec', prov['prior']['sha256'])
cm = json.load(open(R + '/uq/d8/xr_cache/meta.json'))
print('== split sha', sha(cm['split_path']), 'meta', cm['split_sha256'])
sub = json.load(open(REPO + '/tests/data/val_subset_200.json'))
print('subset keys', list(sub.keys()))
f1, f2 = sub['files'], rec['files']
print('n subset', len(f1), 'n rec', len(f2), 'equal list', f1 == f2, 'equal set', set(f1) == set(f2), 'unique rec', len(set(f2)))
print('n classes', len(set(rec['classes'])), 'n source images', len({os.path.dirname(f) + '/' + os.path.basename(f).split('_')[0] for f in f2}))
json.dump(out, open('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/provenance/p2_out.json', 'w'), default=str, indent=1)
