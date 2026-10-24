"""Re-check condition-blindness's C4 (diffusion CRPS vs a constant-width Gaussian around x_r) from its saved per-cube values."""
import json, numpy as np
CB = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
for fn in ('ref200.json', 'ref_c.json'):
    d = json.load(open(CB + fn))
    print(fn, 'keys', list(d.keys())[:30])
    for k, v in d.items():
        if isinstance(v, dict):
            print('  ', k, list(v.keys())[:20])
        elif isinstance(v, list):
            print('  ', k, len(v), v[:2])
        else:
            print('  ', k, v)
