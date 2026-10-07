"""Shared helpers for exp-1d-hascid (HASCID, PH5 sensor, per-pixel U2Net1D).

Everything here is read-only with respect to the repo and the dataset.
"""
import os
import sys
import time
import json
import numpy as np
import torch

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

OLDCODE = os.environ.get('WS_OLDCODE', '/tmp/diffu_oldcode')
CKPT = "/data/chaoyi_he/HSI/Diffu/results/1d_hsi_diffusion/HASCID/PH5/final_model.pth"
DATA = "/data/chaoyi_he/HSI/Diffu/dataset/HASCID-Dataset"
OUT = os.environ.get('WS_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
SNAPSHOT_ORDER = ["50bb95a", "6406385", "988a338"]


def _purge_modules():
    for k in list(sys.modules):
        if k == "model" or k.startswith("model.") or k == "data_loader" or k.startswith("data_loader.") \
                or k == "misc" or k.startswith("misc."):
            del sys.modules[k]


def load_model_strict(verbose=True):
    """Try the snapshots in order; return (model, snapshot, ckpt_meta)."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    errors = {}
    for snap in SNAPSHOT_ORDER:
        d = os.path.join(OLDCODE, snap)
        _purge_modules()
        sys.path.insert(0, d)
        try:
            from model.u2net_1d import U2Net1D
            model = U2Net1D(input_channels=1, condition_dim=30, base_channels=128)
            model.load_state_dict(sd, strict=True)
            for k, v in model.state_dict().items():
                assert torch.equal(v, sd[k]), k
            model.eval()
            meta = {"snapshot": snap, "epoch": ck.get("epoch"), "loss": float(ck.get("loss")),
                    "diffusion_config": {k: (v.tolist() if hasattr(v, "tolist") and getattr(v, "ndim", 0) == 0 else
                                             (str(v) if not hasattr(v, "shape") else f"tensor{tuple(v.shape)}"))
                                         for k, v in ck["diffusion_config"].items()},
                    "n_params": int(sum(p.numel() for p in model.parameters()))}
            if verbose:
                print(f"[load] strict load OK with snapshot {snap}; params={meta['n_params']}")
            dc = ck["diffusion_config"]
            meta["_betas"] = dc["betas"].float().cpu()
            meta["_alpha_bars"] = dc["alpha_bars"].float().cpu()
            meta["_post_variance"] = dc["post_variance"].float().cpu()
            meta["prediction_type"] = dc["prediction_type"]
            del ck
            return model, snap, meta
        except Exception as e:  # noqa
            errors[snap] = repr(e)[:300]
            if verbose:
                print(f"[load] snapshot {snap} failed: {errors[snap]}")
        finally:
            sys.path.remove(d)
    raise RuntimeError(f"no snapshot loads strictly: {errors}")


def get_loader_objects(snap="988a338"):
    """Instantiate the snapshot's HASCID_data for both splits (no data written: resampled_gt exists)."""
    d = os.path.join(OLDCODE, snap)
    _purge_modules()
    sys.path.insert(0, d)
    try:
        from data_loader.my_dataset import HASCID_data
        tr = HASCID_data(DATA, train_mode="pixel", split="train", eval_ratio=0.1, data_format="pixel")
        te = HASCID_data(DATA, train_mode="pixel", split="test", eval_ratio=0.1, data_format="pixel")
    finally:
        sys.path.remove(d)
    return tr, te


class Schedule:
    def __init__(self, betas, alpha_bars, post_variance):
        self.betas = betas.double()
        self.alphas = 1.0 - self.betas
        self.ab = torch.cumprod(self.alphas, 0)
        # sanity: identical to the checkpoint-stored cumulative products
        assert torch.allclose(self.ab.float(), alpha_bars, rtol=1e-5, atol=1e-7)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), self.ab[:-1]])
        self.post_var = self.betas * (1 - ab_prev) / (1 - self.ab)
        assert torch.allclose(self.post_var.float(), post_variance, rtol=1e-4, atol=1e-8)
        self.T = len(betas)


def make_schedule(meta):
    # repo DiffusionTrainer: torch.linspace(1e-4, 0.02, 1000) float32
    b = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float32)
    assert torch.allclose(b, meta["_betas"]), "beta schedule mismatch"
    return Schedule(meta["_betas"], meta["_alpha_bars"], meta["_post_variance"])


def interp_matrix(src_wl, dst_wl):
    """M such that M @ v == interp1d(src_wl, v, linear, extrapolate)(dst_wl) for any v."""
    from scipy.interpolate import interp1d
    eye = np.eye(len(src_wl))
    f = interp1d(src_wl, eye, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(dst_wl)  # [len(dst), len(src)]


def metrics(xhat, x):
    """xhat, x: [N,160] numpy on loader scale [-1,1]. Returns per-pixel rmse(%), sam(deg), psnr(dB) on [0,1]."""
    a = (np.asarray(xhat, np.float64) + 1) / 2
    b = (np.asarray(x, np.float64) + 1) / 2
    mse = ((a - b) ** 2).mean(1)
    rmse = np.sqrt(mse) * 100.0
    psnr = 10 * np.log10(1.0 / np.maximum(mse, 1e-20))
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.clip(num / np.maximum(den, 1e-12), -1, 1)
    sam = np.degrees(np.arccos(cos))
    return rmse, sam, psnr
