"""Phase 2a Task 5: the cache builder refuses a regressor whose train_half does not match its role, centres every
training cube with the OTHER half's regressor and every test-split cube with the full one, stores float16, and writes
sigma_r (RMS over the cross-fitted training residuals) and sigma_full (monitoring files only) into the meta."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_loader.warmstart_paths import file_sha256
from data_loader.xr_cache import cache_path, read_meta
from model.residual_wrapper import build_residual_model
from scripts.uq import build_xr_cache as bx
from scripts.uq.split_halves import write_split

ROOT = '/fake/root'
GATE = ['MatFlower60/Train/P023/0_1.mat']       # one "gate" file in the stub test split (excluded from sigma_full)


class StubDS:
    """(x64, y_d, x0_hat) HWC float32, 16 x 16 cubes (the U-Net pools by 16: 8 x 8 is too small); the residual x - x0_hat
    has RMS 0.1 by construction."""
    sigma_d = 0.1
    prior = {'s': 3e-7}
    d = 4

    def __init__(self, files, seed):
        g = torch.Generator().manual_seed(seed)
        n = len(files)
        self.img_list = [os.path.join(ROOT, f) for f in files]
        self.xh = torch.rand(n, 16, 16, 64, generator=g) * 1.2 - 0.6
        self.x = self.xh + 0.1 * torch.randn(n, 16, 16, 64, generator=g).sign()      # |x - xh| = 0.1 exactly
        self.y = torch.rand(n, 4, 4, 30, generator=g)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


def fake_ckpt(tmp_path, name, train_half, d=4, mode='regress', scale=0.0, split_sha256=None):
    """A regressor checkpoint whose net predicts `scale` everywhere (a constant residual), with the trainer's meta
    (a half regressor's meta also carries the split_sha256 of its split when one is given)."""
    m = build_residual_model('regress', 8)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        last_bias = [p for n, p in m.net.named_parameters() if n.endswith('.bias')][-1]
        last_bias.fill_(scale)                      # zero weights everywhere: the net's output is this constant
    out = m(torch.zeros(1, 64, 16, 16), {'sensor': torch.zeros(1, 30, 4, 4), 'x0_hat': torch.zeros(1, 64, 16, 16)}, torch.zeros(1, dtype=torch.long))
    assert torch.allclose(out, torch.full_like(out, scale)), 'the last .bias of the net is not its output bias; pick the output layer explicitly'
    run = tmp_path / 'runs' / name
    run.mkdir(parents=True)
    path = run / 'checkpoint_epoch_3.pth'
    meta = dict(mode=mode, d=d, base_channels=8, prediction_type='direct', sigma_d=0.1, prior_sha256='p',
                args=dict(train_half=train_half, loss_type='l1', seed=42), train_half=train_half)
    if split_sha256 is not None:
        meta['split_sha256'] = split_sha256
    torch.save(dict(model_state_dict=m.state_dict(), meta=meta), path)
    return str(path)


def test_ckpt_info_checks_role_mode_and_stride(tmp_path):
    sha, other = 'a' * 64, 'b' * 64
    p = fake_ckpt(tmp_path, 'regress_l1_half1_d4_b64', 1, split_sha256=sha)
    ck, info = bx.ckpt_info(p, 'half1', 1, 4)
    assert info['run'] == 'regress_l1_half1_d4_b64' and info['epoch'] == 3 and info['train_half'] == 1 and info['base_channels'] == 8 and len(info['sha256']) == 64
    with pytest.raises(ValueError, match='train_half'):
        bx.ckpt_info(p, 'half2', 2, 4)
    with pytest.raises(ValueError, match='d='):
        bx.ckpt_info(p, 'half1', 1, 8)
    q = fake_ckpt(tmp_path, 'residual_d4_b64', 0, mode='residual')
    with pytest.raises(ValueError, match='mode'):
        bx.ckpt_info(q, 'full', 0, 4)
    # a half regressor must have been trained on the split the builder reads (its meta's split_sha256)
    p2 = fake_ckpt(tmp_path, 'regress_l1_half2_d4_b64', 2, split_sha256=sha)
    for path, role, half in ((p, 'half1', 1), (p2, 'half2', 2)):
        assert bx.ckpt_info(path, role, half, 4, split_sha=sha)[1]['train_half'] == half
        with pytest.raises(ValueError, match='split_sha256'):
            bx.ckpt_info(path, role, half, 4, split_sha=other)
    nosha = fake_ckpt(tmp_path, 'regress_l1_half1_nosha', 1)
    with pytest.raises(ValueError, match='split_sha256'):
        bx.ckpt_info(nosha, 'half1', 1, 4, split_sha=sha)
    full = fake_ckpt(tmp_path, 'regress_l1_sel_d4_b64', 0)                 # the full regressor names no split
    assert bx.ckpt_info(full, 'full', 0, 4, split_sha=sha)[1]['train_half'] == 0


def test_half_of():
    assert bx.half_of('a', {'a'}, {'b'}) == 1 and bx.half_of('b', {'a'}, {'b'}) == 2
    with pytest.raises(KeyError):
        bx.half_of('c', {'a'}, {'b'})


def test_run_crossfits_train_uses_full_on_val_and_writes_sigmas(tmp_path, monkeypatch):
    train_files = [f'MatFlower60/Train/P00{1 + i % 2}/{i}_1.mat' for i in range(6)]
    val_files = GATE + ['MatFlower60/Train/P023/1_1.mat', 'MatFlower60/Train/P024/2_1.mat']
    split = dict(seed=0, half1=sorted(train_files[0::2]), half2=sorted(train_files[1::2]), images_per_class={})
    sp = tmp_path / 'split.json'
    write_split(str(sp), split)
    sha = file_sha256(str(sp))
    # half1's net sees H1 -> must centre H2; give the three nets different constant residuals to tell them apart
    h1 = fake_ckpt(tmp_path, 'regress_l1_half1_d4_b64', 1, scale=1.0, split_sha256=sha)
    h2 = fake_ckpt(tmp_path, 'regress_l1_half2_d4_b64', 2, scale=2.0, split_sha256=sha)
    fu = fake_ckpt(tmp_path, 'regress_l1_sel_d4_b64', 0, scale=3.0)
    monkeypatch.setattr(bx, 'gate_relpaths', lambda data_root: set(GATE))
    ds = {'train': StubDS(train_files, 0), 'val': StubDS(val_files, 1)}
    out = str(tmp_path / 'xr_cache')
    args = SimpleNamespace(d=4, half1=h1, half2=h2, full=fu, out=out, split_json=str(sp), device='cpu', batch_size=4,
                           num_workers=0, limit_files=0, data_root=ROOT, prior='unused')
    # run() checks each half regressor against the sha256 of the split it reads, before writing anything
    stale = fake_ckpt(tmp_path, 'regress_l1_half1_stale', 1, scale=1.0, split_sha256='c' * 64)
    with pytest.raises(ValueError, match='split_sha256'):
        bx.run(SimpleNamespace(**{**vars(args), 'half1': stale}), make_ds=lambda split_name: ds[split_name])
    assert not os.path.exists(out)
    meta = bx.run(args, make_ds=lambda split_name: ds[split_name])
    assert read_meta(out) == meta and meta['n_train'] == 6 and meta['n_val'] == 3 and meta['n_val_monitoring'] == 2
    assert meta['half1']['train_half'] == 1 and meta['half2']['train_half'] == 2 and meta['full']['train_half'] == 0
    assert meta['split_sha256'] == sha
    for i, f in enumerate(train_files):
        xr = np.load(cache_path(out, 'train', f))
        want_scale = 2.0 if f in split['half1'] else 1.0              # H1 cubes are centred by the half2 net
        expect = np.clip(ds['train'].xh[i].numpy() + 0.1 * want_scale, -1, 1)
        assert xr.dtype == np.float16 and np.abs(xr.astype(np.float32) - expect).max() < 2e-3
    for i, f in enumerate(val_files):
        xr = np.load(cache_path(out, 'val', f))
        expect = np.clip(ds['val'].xh[i].numpy() + 0.1 * 3.0, -1, 1)
        assert np.abs(xr.astype(np.float32) - expect).max() < 2e-3
    # sigma_r: RMS of x - x_r over the train cubes (float32 x_r), computed independently here
    ss, n = 0.0, 0
    for i, f in enumerate(train_files):
        want_scale = 2.0 if f in split['half1'] else 1.0
        xr = np.clip(ds['train'].xh[i].numpy() + 0.1 * want_scale, -1, 1)
        ss += float(((ds['train'].x[i].numpy() - xr) ** 2).sum()); n += xr.size
    assert abs(meta['sigma_r'] - np.sqrt(ss / n)) < 1e-6
    assert meta['sigma_flag'] == (not 0.9 <= meta['sigma_ratio'] <= 1.1) and meta['dtype'] == 'float16'
    with pytest.raises(FileExistsError):
        bx.run(args, make_ds=lambda split_name: ds[split_name])
