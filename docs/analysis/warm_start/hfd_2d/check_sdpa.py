# CPU float32 check that the SDPA attention patch is numerically the original attention.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
torch.set_num_threads(16)
model, betas, meta = load_model(torch.device('cpu'), sdpa=False)
# float32 on CPU (the time embedding is hard-coded float32; CPU has no TF32)
tr, ev = make_datasets()
a, b = ev[0]
x = torch.tensor(a).permute(2, 0, 1)[None].float()
y = torch.tensor(b).permute(2, 0, 1)[None].float()
g = torch.Generator().manual_seed(0)
eps = torch.randn(x.shape, generator=g, dtype=torch.float32)
ab = torch.cumprod(1 - betas, 0).float()
t = torch.tensor([500])
xt = ab[t].sqrt() * x + (1 - ab[t]).sqrt() * eps
with torch.no_grad():
    e0 = model(xt, y, t)
    patch_attention_sdpa()
    e1 = model(xt, y, t)
print('float32 CPU: max|sdpa - orig| =', (e1 - e0).abs().max().item(), ' rms eps =', e0.pow(2).mean().sqrt().item())
print('eps-MSE vs true eps at t=500:', (e0 - eps).pow(2).mean().item())
