import os
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d4'
for root, dirs, files in os.walk(R):
    if 'xr_cache' in root:
        continue
    for f in files:
        p = os.path.join(root, f)
        print(os.path.getsize(p), p)
