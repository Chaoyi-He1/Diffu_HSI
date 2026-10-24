import os, glob
R='/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq'
for root, dirs, files in os.walk(R):
    if 'xr_cache' in root and root.count('/')>R.count('/')+2: continue
    if len(files) > 20:
        print(root, len(files), 'files')
        continue
    for f in files:
        p=os.path.join(root,f); print(p, os.path.getsize(p))
P='/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/docs/superpowers/plans'
print(os.listdir(P))
for f in glob.glob(P+'/2026-10-22*'):
    txt=open(f).read().splitlines()
    for i,l in enumerate(txt):
        if 'ddpm' in l.lower() or 'fix round' in l.lower():
            print(os.path.basename(f), i+1, l[:300])
