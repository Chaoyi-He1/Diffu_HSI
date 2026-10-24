import json, numpy as np
r = json.load(open('/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8/eval/d8_main.json'))
def walk(o, p='', depth=0):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict,)) and depth < 4:
                print('  '*depth + f'{k}: dict({len(v)})')
                walk(v, p+'.'+k, depth+1)
            elif isinstance(v, list):
                el = v[0] if v else None
                print('  '*depth + f'{k}: list[{len(v)}] first={str(el)[:100]}')
            else:
                print('  '*depth + f'{k}: {str(v)[:150]}')
walk(r)
