import json, sys
import numpy as np
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
which = sys.argv[1] if len(sys.argv) > 1 else 'd4'
paths = {
    'd4': R + 'd4/eval/d4_contrast.json',
    'd8': R + 'd8/eval/d8_main.json',
    'd8k64': R + 'd8/eval/d8_fix_k64.json',
    'v4': R + 'd4/verdict__d4_contrast__report.json',
    'v8': R + 'd8/verdict__d8_main.json',
    'v8k64': R + 'd8/verdict__d8_fix_k64.json',
}
rec = json.load(open(paths[which]))


def show(d, pre=''):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(pre + str(k) + ':{')
                show(v, pre + '  ')
            elif isinstance(v, list):
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    print(pre + str(k), 'list[%d] num' % len(v), v[:3])
                elif len(v) > 0 and isinstance(v[0], str):
                    print(pre + str(k), 'list[%d] str' % len(v), v[:2])
                else:
                    print(pre + str(k), 'list[%d]' % len(v), str(v[:1])[:200])
            else:
                print(pre + str(k), repr(v)[:300])


show(rec)
