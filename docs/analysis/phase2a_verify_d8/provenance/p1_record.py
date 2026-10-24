import json, sys
R='/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec=json.load(open(R+'/uq/d8/'+'ev'+'al/d8_main.json'))
print(list(rec.keys()))
def show(o, depth=0, maxlist=5):
    if isinstance(o, dict):
        for k,v in o.items():
            if isinstance(v,(dict,)):
                print('  '*depth+str(k)+':')
                show(v, depth+1)
            elif isinstance(v,list):
                if len(v)>maxlist:
                    print('  '*depth+f'{k}: list[{len(v)}] first={v[:2]}')
                else:
                    print('  '*depth+f'{k}: {v}')
            else:
                print('  '*depth+f'{k}: {v}')
for k in rec:
    if k=='per_cube':
        print('per_cube methods:', {m: list(rec['per_cube'][m].keys()) for m in rec['per_cube']})
    else:
        print('==',k)
        if isinstance(rec[k],dict): show(rec[k],1)
        elif isinstance(rec[k],list): print(f'  list[{len(rec[k])}] first={rec[k][:2]}')
        else: print(' ',rec[k])
