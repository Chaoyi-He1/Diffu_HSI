import os, sys, time
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import torch
torch.set_num_threads(int(os.environ.get('NT', '8')))
from model.residual_wrapper import build_residual_model
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
t0 = time.time()
ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_100.pth', map_location='cpu', weights_only=False)
print('keys', list(ck.keys()), 'load', time.time() - t0)
meta = ck['meta']
print({k: v for k, v in meta.items() if k not in ('args',)})
m = build_residual_model(meta['mode'], int(meta['base_channels']))
m.load_state_dict(ck['model_state_dict'], strict=True)
m.eval()
print('params', sum(p.numel() for p in m.parameters()) / 1e6)
for B in (1, 4):
    xt = torch.randn(B, 64, 64, 64)
    cond = {'sensor': torch.randn(B, 30, 8, 8), 'x0_hat': torch.randn(B, 64, 64, 64)}
    t = torch.full((B,), 500, dtype=torch.long)
    with torch.no_grad():
        for _ in range(2):
            t1 = time.time(); out = m(xt, cond, t); print(B, 'fwd', time.time() - t1, out.shape)
