# Verify which code snapshot loads the HFD checkpoint with strict=True.
# Usage: python check_load.py <snapshot_dir>
import os, sys, time
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import torch
CKPT = '/data/chaoyi_he/HSI/Diffu/results/2d_hsi_diffusion/HFD/R_1/l2_loss/checkpoint_epoch_101.pth'
snap = sys.argv[1]
sys.path.insert(0, snap)
from model.u2net_hyperspectral import U2NetHyperspectral
t = time.time()
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
print('loaded in', round(time.time() - t, 1), 's; keys', list(ck.keys()))
print('epoch', ck.get('epoch'), 'loss', ck.get('loss'))
dc = ck.get('diffusion_config')
if dc:
    for k, v in dc.items():
        if torch.is_tensor(v):
            print(' dc', k, tuple(v.shape), v.flatten()[:3].tolist(), v.flatten()[-2:].tolist())
        else:
            print(' dc', k, v)
sd = ck['model_state_dict']
print('input_proj.weight', tuple(sd['input_proj.weight'].shape), 'final.weight', tuple(sd['final.weight'].shape))
m = U2NetHyperspectral(spectral_channels=64, sensor_channels=30, base_channels=128)
res = m.load_state_dict(sd, strict=True)
print('strict load OK:', res)
print('params', sum(p.numel() for p in m.parameters()))
