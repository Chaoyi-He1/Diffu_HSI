"""Shared helpers for the truncated-sampling ("warm start") experiment on the
HFD 64-band U2NetHyperspectral checkpoint (epoch 101, eps-prediction, R_Device1).

All model / data code is imported from the code snapshot 988a338 (2025-10-06),
which loads the checkpoint with strict=True.  Nothing in the repo is modified.
"""
import os, sys, time, json, math
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
# shared server: keep BLAS thread pools small (set before numpy/torch are imported)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
import numpy as np
import torch
import torch.nn.functional as F

SNAP = os.path.join(os.environ.get('WS_OLDCODE', '/tmp/diffu_oldcode'), '988a338')
if SNAP not in sys.path:
    sys.path.insert(0, SNAP)

CKPT = '/data/chaoyi_he/HSI/Diffu/results/2d_hsi_diffusion/HFD/R_1/l2_loss/checkpoint_epoch_101.pth'
DATA = '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset'
OUT = os.environ.get('WS_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
GPU_CAP_BYTES = int(2.4 * 1024 ** 3)   # stay under 2.5 GB on the shared GPU 0


def gpu0_free_mib():
    """Free memory on GPU 0 as reported by nvidia-smi, read BEFORE creating a CUDA context."""
    import subprocess
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        i, used, tot = [int(v) for v in line.split(',')]
        if i == 0:
            return tot - used
    return 0


def pick_device():
    """GPU 0 only if nvidia-smi shows >= 3072 MiB free; then cap this process at 2.4 GiB."""
    free = gpu0_free_mib()
    print('nvidia-smi GPU0 free MiB:', free, flush=True)
    if free >= 3072 and torch.cuda.is_available():
        dev = torch.device('cuda:0')
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(GPU_CAP_BYTES / tot, 0)
        return dev
    print('GPU 0 does not qualify -> CPU', flush=True)
    return torch.device('cpu')


# --------------------------------------------------------------------------
# data: use the snapshot's own HFD_data class so preprocessing is identical
# --------------------------------------------------------------------------
def make_datasets():
    """Return (train_ds, eval_ds) built exactly as the training run built them:
    HFD_data(type='Flower', R_n=1, eval_ratio=0.1, data_format='image'), all files
    (no 10 000-file cap in this snapshot), full-resolution sensor."""
    from data_loader.HFD_dataset import HFD_data
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tr = HFD_data(DATA, train_mode='image', eval_ratio=0.1, split='train', data_format='image', type='Flower', R_n=1)
        ev = HFD_data(DATA, train_mode='image', eval_ratio=0.1, split='test', data_format='image', type='Flower', R_n=1)
    return tr, ev


def class_of(path):
    return os.path.basename(os.path.dirname(path))


def interp_matrix(wl31, n_new=64):
    """64x31 matrix A with x64 = A @ x31, identical to HFD_data.expand_wavelens
    (scipy interp1d, kind='linear', on np.linspace(451, 855, 64))."""
    from scipy.interpolate import interp1d
    new = np.linspace(wl31[0], wl31[-1], n_new)
    A = np.zeros((n_new, len(wl31)))
    I = np.eye(len(wl31))
    for j in range(len(wl31)):
        A[:, j] = interp1d(wl31, I[j], kind='linear', bounds_error=False, fill_value='extrapolate')(new)
    return A


def load_cube31(path):
    """31-band cube on the loader's [-1,1] scale (same lines as HFD_data.__getitem__)."""
    import scipy.io as sio
    gt = np.array(sio.loadmat(path)['truth'], dtype=np.float32)
    mn, mx = gt.min(), gt.max()
    gt = (gt - mn) / (mx - mn + 1e-20)
    return gt * 2.0 - 1.0


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def patch_attention_sdpa():
    """Replace the einsum/softmax attention of CrossAttention.forward with
    torch's fused scaled_dot_product_attention.  Same math (scale = d_head**-0.5,
    no mask is ever passed, dropout p=0 in eval), but O(N) memory, which is what
    lets 64x64 inputs at batch 4-8 fit in the 2.5 GB budget."""
    from model import layers

    def forward(self, x, context=None, mask=None):
        assert mask is None
        h = self.heads
        if context is None:
            context = x
        q = self.to_q(x); k = self.to_k(context); v = self.to_v(context)
        q, k, v = map(lambda t: t.view(*t.shape[:-1], h, self.d_head).transpose(-2, -3), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(-2, -3).reshape(*x.shape[:-1], h * self.d_head)
        return self.to_out(out)

    layers.CrossAttention._orig_forward = layers.CrossAttention.forward
    layers.CrossAttention.forward = forward


def load_model(device, sdpa=True):
    from model.u2net_hyperspectral import U2NetHyperspectral
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    dc = ck['diffusion_config']
    assert dc['prediction_type'] == 'eps' and dc['n_timesteps'] == 1000
    m = U2NetHyperspectral(spectral_channels=64, sensor_channels=30, base_channels=128)
    m.load_state_dict(ck['model_state_dict'], strict=True)
    m.eval()
    if sdpa:
        patch_attention_sdpa()
    betas = dc['betas'].detach().cpu().double()
    meta = dict(epoch=ck['epoch'], loss=ck['loss'], prediction_type=dc['prediction_type'],
                n_timesteps=dc['n_timesteps'], loss_type=dc['loss_type'],
                beta_first=float(betas[0]), beta_last=float(betas[-1]))
    del ck
    return m.to(device), betas, meta


class Schedule:
    """Linear-beta DDPM schedule (identical numbers to the checkpoint's diffusion_config)."""
    def __init__(self, betas, device):
        b = betas.double()
        ref = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float32).double()
        assert torch.allclose(b, ref, atol=1e-7), 'checkpoint betas are not linear 1e-4..0.02'
        self.betas = b
        self.alphas = 1 - b
        self.ab = torch.cumprod(self.alphas, 0)
        self.ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), self.ab[:-1]])
        self.post_var = b * (1 - self.ab_prev) / (1 - self.ab)
        self.device = device

    def f(self, v):
        return float(v)


# --------------------------------------------------------------------------
# metrics on the [0,1] display scale
# --------------------------------------------------------------------------
def metrics(xhat, x):
    """xhat, x: [B,64,H,W] on [-1,1].  Returns per-item dict lists (rmse %, sam deg, psnr)."""
    a = ((xhat.clamp(-1, 1) + 1) / 2).double()
    b = ((x + 1) / 2).double()
    mse = ((a - b) ** 2).flatten(1).mean(1)
    rmse = torch.sqrt(mse) * 100.0
    psnr = 10 * torch.log10(1.0 / mse)
    dot = (a * b).sum(1)
    na = a.norm(dim=1); nb = b.norm(dim=1)
    cos = (dot / (na * nb).clamp_min(1e-12)).clamp(-1, 1)
    sam = torch.rad2deg(torch.acos(cos)).flatten(1).mean(1)
    return rmse.cpu().numpy(), sam.cpu().numpy(), psnr.cpu().numpy()
