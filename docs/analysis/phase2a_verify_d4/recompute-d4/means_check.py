import json

import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
for p in ('d4/eval/d4_contrast.json', 'd8/eval/d8_main.json'):
    rec = json.load(open(R + p))
    mx = 0.0
    lens = set()
    for m, d in rec['per_cube'].items():
        for k, v in d.items():
            lens.add(len(v))
            mx = max(mx, abs(float(np.mean(v)) - rec['means'][m][k]))
            if not np.all(np.isfinite(v)):
                print('NONFINITE', p, m, k)
    print(p, 'max |mean(per_cube) - means| =', mx, 'array lengths', lens, 'K', rec['K'], 'coverage_exact', rec['coverage_exact'],
          'centre', rec['provenance']['centre']['run'], rec['provenance']['centre']['epoch'])
    # varhead mean metrics identical to regressor (same centre)?
    print('  varhead mean_rmse == regressor rmse:', rec['per_cube']['varhead']['mean_rmse_pct'] == rec['per_cube']['regressor']['rmse_pct'])
