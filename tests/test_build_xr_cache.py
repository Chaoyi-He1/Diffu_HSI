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
TRAIN = [f'MatFlower60/Train/P00{1 + i % 2}/{i}_1.mat' for i in range(6)]
VAL = GATE + ['MatFlower60/Train/P023/1_1.mat', 'MatFlower60/Train/P024/2_1.mat']


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


def fake_ckpt(tmp_path, name, train_half, d=4, mode='regress', scale=0.0, split_sha256=None, sigma_d=0.1, prior_sha256='p'):
    """A regressor checkpoint whose net predicts `scale` everywhere (a constant residual), with the trainer's meta
    (split_sha256 and prior_sha256 only when not None, so a test can leave either key out)."""
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
    meta = dict(mode=mode, d=d, base_channels=8, prediction_type='direct', sigma_d=sigma_d,
                args=dict(train_half=train_half, loss_type='l1', seed=42), train_half=train_half)
    for key, value in (('split_sha256', split_sha256), ('prior_sha256', prior_sha256)):
        if value is not None:
            meta[key] = value
    torch.save(dict(model_state_dict=m.state_dict(), meta=meta), path)
    return str(path)


def setup_run(tmp_path, monkeypatch, half1=None, half2=None, prior='unused', prior_sha256='p'):
    """A split file (default: TRAIN's even files in half1, odd ones in half2), three regressors naming it whose nets
    predict the constant residuals 1 (half1), 2 (half2) and 3 (full) so the test can tell them apart, the stub datasets
    and the builder's args. Returns (args, split, the split's sha256, datasets)."""
    split = dict(seed=0, half1=sorted(TRAIN[0::2]) if half1 is None else half1,
                 half2=sorted(TRAIN[1::2]) if half2 is None else half2, images_per_class={})
    sp = tmp_path / 'split.json'
    write_split(str(sp), split)
    sha = file_sha256(str(sp))
    h1 = fake_ckpt(tmp_path, 'regress_l1_half1_d4_b64', 1, scale=1.0, split_sha256=sha, prior_sha256=prior_sha256)
    h2 = fake_ckpt(tmp_path, 'regress_l1_half2_d4_b64', 2, scale=2.0, split_sha256=sha, prior_sha256=prior_sha256)
    fu = fake_ckpt(tmp_path, 'regress_l1_sel_d4_b64', 0, scale=3.0, prior_sha256=prior_sha256)
    monkeypatch.setattr(bx, 'gate_relpaths', lambda data_root: set(GATE))
    ds = {'train': StubDS(TRAIN, 0), 'val': StubDS(VAL, 1)}
    args = SimpleNamespace(d=4, half1=h1, half2=h2, full=fu, out=str(tmp_path / 'xr_cache'), split_json=str(sp),
                           device='cpu', batch_size=4, num_workers=0, limit_files=0, data_root=ROOT, prior=prior)
    return args, split, sha, ds


def run_with(args, ds, **override):
    return bx.run(SimpleNamespace(**{**vars(args), **override}), make_ds=lambda split_name: ds[split_name])


def rms64(xs, xrs):
    """RMS of x - x_r over a list of cubes: the float32 difference, squared and summed in float64."""
    ss = sum(float(((x - xr).astype(np.float64) ** 2).sum()) for x, xr in zip(xs, xrs))
    return float(np.sqrt(ss / sum(x.size for x in xs)))


def test_ckpt_info_checks_role_mode_and_stride(tmp_path):
    sha, other = 'a' * 64, 'b' * 64
    p = fake_ckpt(tmp_path, 'regress_l1_half1_d4_b64', 1, split_sha256=sha)
    ck, info = bx.ckpt_info(p, 'half1', 1, 4)
    assert info['run'] == 'regress_l1_half1_d4_b64' and info['epoch'] == 3 and info['train_half'] == 1 and info['base_channels'] == 8 and len(info['sha256']) == 64
    assert info['sigma_d'] == 0.1 and info['prior_sha256'] == 'p'
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
    args, split, sha, ds = setup_run(tmp_path, monkeypatch)
    out = args.out
    # run() checks each half regressor against the sha256 of the split it reads, before writing anything
    stale = fake_ckpt(tmp_path, 'regress_l1_half1_stale', 1, scale=1.0, split_sha256='c' * 64)
    with pytest.raises(ValueError, match='split_sha256'):
        run_with(args, ds, half1=stale)
    assert not os.path.exists(out)
    meta = run_with(args, ds)
    assert read_meta(out) == meta and meta['n_train'] == 6 and meta['n_val'] == 3 and meta['n_val_monitoring'] == 2
    assert meta['half1']['train_half'] == 1 and meta['half2']['train_half'] == 2 and meta['full']['train_half'] == 0
    assert meta['split_sha256'] == sha and meta['prior_sha256'] is None          # args.prior 'unused' does not exist
    train_xr, val_xr = [], []
    for i, f in enumerate(TRAIN):
        xr = np.load(cache_path(out, 'train', f))
        want_scale = 2.0 if f in split['half1'] else 1.0              # H1 cubes are centred by the half2 net
        expect = np.clip(ds['train'].xh[i].numpy() + 0.1 * want_scale, -1, 1)
        assert xr.dtype == np.float16 and np.abs(xr.astype(np.float32) - expect).max() < 2e-3
        train_xr.append(expect)
    for i, f in enumerate(VAL):
        xr = np.load(cache_path(out, 'val', f))
        expect = np.clip(ds['val'].xh[i].numpy() + 0.1 * 3.0, -1, 1)
        assert np.abs(xr.astype(np.float32) - expect).max() < 2e-3
        val_xr.append(expect)
    # sigma_r over the cross-fitted train cubes, sigma_full over the monitoring val files (VAL[0] is the gate file), both
    # from the float32 x_r before the float16 cast, computed independently here
    assert abs(meta['sigma_r'] - rms64([ds['train'].x[i].numpy() for i in range(len(TRAIN))], train_xr)) < 1e-9
    assert abs(meta['sigma_full'] - rms64([ds['val'].x[i].numpy() for i in range(1, len(VAL))], val_xr[1:])) < 1e-9
    assert meta['sigma_ratio'] == pytest.approx(meta['sigma_full'] / meta['sigma_r'])
    assert meta['sigma_flag'] == (not 0.9 <= meta['sigma_ratio'] <= 1.1) and meta['sigma_flag'] is True   # ratio ~ 1.69
    assert meta['dtype'] == 'float16'
    with pytest.raises(FileExistsError):
        run_with(args, ds)


@pytest.mark.parametrize('role', ['half2', 'full'])
def test_run_refuses_a_regressor_whose_sigma_d_is_not_the_datasets(tmp_path, monkeypatch, role):
    args, _, sha, ds = setup_run(tmp_path, monkeypatch)
    half = dict(half2=2, full=0)[role]
    bad = fake_ckpt(tmp_path, f'regress_l1_{role}_other', half, scale=1.0, split_sha256=sha if half else None, sigma_d=0.2)
    with pytest.raises(ValueError, match=r'sigma_d 0\.2'):
        run_with(args, ds, **{role: bad})
    assert not os.path.exists(args.out)          # both splits' datasets are checked before the first cube is written


def test_run_checks_each_regressors_prior_sha256_against_the_prior_file(tmp_path, monkeypatch):
    prior = tmp_path / 'prior.npz'
    prior.write_bytes(b'any bytes stand in for the prior')
    psha = file_sha256(str(prior))
    args, _, sha, ds = setup_run(tmp_path, monkeypatch, prior=str(prior), prior_sha256=psha)
    other = fake_ckpt(tmp_path, 'regress_l1_sel_other', 0, scale=3.0, prior_sha256='q' * 64)
    missing = fake_ckpt(tmp_path, 'regress_l1_half1_missing', 1, scale=1.0, split_sha256=sha, prior_sha256=None)
    with pytest.raises(ValueError, match="prior_sha256 'q{64}'"):
        run_with(args, ds, full=other)
    with pytest.raises(ValueError, match='prior_sha256 None'):
        run_with(args, ds, half1=missing)
    assert not os.path.exists(args.out)
    meta = run_with(args, ds)
    assert meta['prior_sha256'] == psha and meta['prior_path'] == str(prior) and meta['full']['prior_sha256'] == psha


@pytest.mark.parametrize('half1, half2, msg', [(sorted(TRAIN[0::2] + TRAIN[1:2]), sorted(TRAIN[1::2]), 'both halves'),
                                               (sorted(TRAIN[0::2]), sorted(TRAIN[1::2])[:-1], 'neither half')])
def test_run_refuses_a_split_with_shared_or_missing_files(tmp_path, monkeypatch, half1, half2, msg):
    args, _, _, ds = setup_run(tmp_path, monkeypatch, half1=half1, half2=half2)
    with pytest.raises(ValueError, match=msg):
        run_with(args, ds)
    assert not os.path.exists(args.out)
