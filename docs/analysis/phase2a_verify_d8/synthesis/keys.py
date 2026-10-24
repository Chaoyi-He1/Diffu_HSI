import json
R='/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec=json.load(open(R+'/uq/d8/eval/d8_main.json'))
def walk(d,p='',depth=0):
    if depth>2: return
    for k,v in d.items():
        if isinstance(v,dict):
            print(p+k+'/'); walk(v,p+'  ',depth+1)
        elif isinstance(v,list):
            print(p+k, 'list', len(v), (v[:3] if len(v) and not isinstance(v[0],(list,dict)) else ''))
        else:
            print(p+k, repr(v)[:150])
walk({k:v for k,v in rec.items() if k!='per_cube'})
print('per_cube methods', {m:list(v.keys()) for m,v in rec['per_cube'].items()})
v=json.load(open(R+'/uq/d8/verdict__d8_main.json'))
print(json.dumps(v,indent=1)[:5000])
