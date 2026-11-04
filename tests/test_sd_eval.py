"""Phase 3 Task 10: sd_eval scores L / L_demosaic / D / V / C / C1 / B per cube into one record, refuses checkpoints of
another config sha or mode, refuses to overwrite; stub models and a stub sensor dataset on CPU."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_loader.hfd_sensor_sim import sensor_collate_fn
from data_loader.linear_estimate import expand_matrix
from data_loader.mmse import mmse_maps
from data_loader.sensor_config import default_training_fields, training_sha
from data_loader.sensor_forward import measure, sigma_from_snr
from model.residual_wrapper import build_residual_model
from scripts.sensor import sd_eval as SE

ROOT = '/fake/root'
H = 8
SIGMA_D = 0.02
CFG = dict(default_training_fields('prior'), P_y=1.0, e2_demosaic=[0.001] * 30, devices={}, n_files_P_y=1,
           created='now', prior_path='p')


class StubSensorDS(torch.utils.data.Dataset):
    """Task 5's dict items for a fixed (mode, SNR); `corrupt` adds 1e3 to y where the mask is 0."""

    def __init__(self, R, prior, mode='sequential', snr_db=30.0, n=5, corrupt=0.0):
        self.R, self.A = R, expand_matrix(np.linspace(451, 855, 31), 64)
        self.prior = dict(prior, s=np.sqrt(np.diag(prior['C'])))
        self.sigma_d = SIGMA_D
        self.mode, self.snr_db, self.corrupt = mode, snr_db, corrupt
        self.img_list = [os.path.join(ROOT, f'MatFlower60/Train/P02{2 + i % 3}/{i}_1.mat') for i in range(n)]
        self.sensor = {'sha256': 'r'}
        self.config_sha = training_sha(CFG)
        self.collate_fn = sensor_collate_fn
        X = prior['X']
        rng = np.random.default_rng(3)
        self.cubes = [np.clip(0.5 + X[rng.integers(0, len(X), H * H)].reshape(H, H, 31), 0, 1) for _ in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        x01 = self.cubes[i]
        sigma = sigma_from_snr(self.snr_db, CFG['P_y'])
        y, mask = measure(x01, self.R, self.mode, sigma, np.random.default_rng(i))
        snap = self.mode == 'snapshot'
        xh, s = mmse_maps(y, mask if snap else None, self.R, self.prior, sigma, self.A)
        y = (y + self.corrupt * (1.0 - mask)).astype(np.float32)
        x = ((2.0 * x01 - 1.0) @ self.A.T).astype(np.float32)
        return dict(x=x, y=y, x0_hat=xh, s_post=np.maximum(s, SIGMA_D).astype(np.float32), s_raw=s, mask=mask,
                    sigma=np.float32(sigma), snr_db=np.float32(self.snr_db), snapshot=np.float32(snap))


class Const(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x_t, cond, t):
        return torch.full_like(x_t, self.c)


class Noisy(torch.nn.Module):
    """A v-prediction net that depends on x_t, so DDIM draws differ across seeds."""
    def forward(self, x_t, cond, t):
        return 0.1 * x_t


def fake_args(**kw):
    a = SimpleNamespace(name='t', sensor_device='device1', mode='sequential', snr_db=30.0, subset='gate', methods=None,
                        regress=None, varhead=None, cond=None, uncond=None, K=3, pigdm_steps=2, guidance='pigdm',
                        sensor_config='cfg.json', batch_size=2, num_workers=0, limit_files=0, device='cpu',
                        data_root=ROOT, prior='p')
    a.__dict__.update(kw)
    return a


def run(ds, args, models):
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=ds.collate_fn)
    return SE.run_eval(models, ds, loader, args, CFG, {}, torch.device('cpu'))


def save_ckpt(path, mode, sha=None):
    torch.save(dict(model_state_dict=build_residual_model(mode, 16).state_dict(),
                    meta=dict(mode=mode, base_channels=16, sensor_config_sha=sha or training_sha(CFG), prediction_type='v')),
               path)
    return str(path)


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')                # K = 3 < min_k_for(0.9)
def test_record_holds_every_requested_method_with_paired_files(small_R, synthetic_prior):
    ds = StubSensorDS(small_R, synthetic_prior)
    methods = ['L', 'D', 'V', 'C', 'C1', 'B']
    models = dict(D=Const(0.0), V=Const(0.0), C=Noisy(), B=Noisy())
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = run(ds, fake_args(methods=methods), models)
    assert set(rec['per_cube']) == set(methods) and 'L_demosaic' not in rec['per_cube']
    for m, d in rec['per_cube'].items():
        for k, v in d.items():
            assert len(v) == rec['n'] == 5, (m, k)
    assert 'rmse_pct' in rec['per_cube']['C1'] and 'crps' in rec['per_cube']['C'] and 'crps' in rec['per_cube']['B']
    assert 'crps' in rec['per_cube']['L'] and 'cov90' in rec['per_cube']['V']
    # D with a zero residual is the clamped MMSE centre (compose clamps to [-1, 1])
    b = sensor_collate_fn([ds[i] for i in range(len(ds))])
    assert rec['per_cube']['D']['rmse_pct'] == SE.point_metrics(b['x0_hat'].clamp(-1, 1), b['x'])['rmse_pct']
    assert rec['files'] == [os.path.relpath(f, ROOT) for f in ds.img_list]
    assert rec['classes'] == ['P022', 'P023', 'P024', 'P022', 'P023']
    s = rec['setting']
    assert s['sensor_device'] == 'device1' and s['mode'] == 'sequential' and s['snr_db'] == 30.0 and s['R_sha256'] == 'r'
    assert abs(s['sigma'] - sigma_from_snr(30.0, 1.0)) < 1e-6
    assert set(rec) == {'name', 'setting', 'subset', 'K', 'pigdm_steps', 'guidance', 'n', 'files', 'classes', 'per_cube',
                        'means', 'coverage_exact', 'provenance'}
    assert set(rec['provenance']) == {'sensor_config_sha', 'config_path', 'checkpoints', 'git_head', 'created', 'seconds',
                                      'seeds', 'device', 'batch_size', 'limit_files'}
    assert rec['provenance']['sensor_config_sha'] == training_sha(CFG) and rec['provenance']['seeds'] == [0, 1, 2]
    json.dumps(rec)


def test_snapshot_adds_L_demosaic(small_R, synthetic_prior):
    ds = StubSensorDS(small_R, synthetic_prior, mode='snapshot', n=3)
    args = fake_args(mode='snapshot')
    assert SE.resolve_methods(args) == ['L', 'L_demosaic']
    assert SE.resolve_methods(fake_args()) == ['L']
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = run(ds, args, {})
    assert set(rec['per_cube']) == {'L', 'L_demosaic'} and len(rec['per_cube']['L_demosaic']['mean_rmse_pct']) == 3
    assert 'crps' in rec['per_cube']['L_demosaic'] and rec['setting']['mode'] == 'snapshot'
    with pytest.raises(ValueError, match='L_demosaic'):
        SE.resolve_methods(fake_args(methods=['L', 'L_demosaic']))         # sequential: no mosaic to demosaic


def test_refuses_checkpoint_with_other_config_sha(tmp_path):
    good = save_ckpt(tmp_path / 'checkpoint_epoch_7.pth', 'regress_fam')
    models, prov = SE.check_models(fake_args(methods=['L', 'D'], regress=good), CFG, torch.device('cpu'))
    assert set(models) == {'D'} and prov['regress']['epoch'] == 7 and prov['regress']['mode'] == 'regress_fam'
    assert set(prov['regress']) == {'path', 'epoch', 'sha256', 'mode', 'git_head'}
    bad = save_ckpt(tmp_path / 'other.pth', 'regress_fam', sha='other')
    with pytest.raises(ValueError, match='other.pth'):
        SE.check_models(fake_args(methods=['L', 'D'], regress=bad), CFG, torch.device('cpu'))
    with pytest.raises(ValueError, match='--regress'):                    # D requested without its checkpoint
        SE.check_models(fake_args(methods=['L', 'D']), CFG, torch.device('cpu'))


def test_refuses_wrong_mode_checkpoint(tmp_path):
    p = save_ckpt(tmp_path / 'reg.pth', 'regress_fam')
    with pytest.raises(ValueError, match='reg.pth.*cond_fam'):
        SE.check_models(fake_args(methods=['C'], cond=p), CFG, torch.device('cpu'))


def test_refuses_overwrite(tmp_path, monkeypatch):
    p = str(tmp_path / 'eval' / 'x.json')
    SE.write_record(p, dict(a=1))
    with open(p) as f:
        assert json.load(f) == dict(a=1)
    with pytest.raises(FileExistsError):
        SE.write_record(p, dict(a=2))
    monkeypatch.setattr(SE, 'output_path', lambda name: p)
    monkeypatch.setattr(sys, 'argv', ['sd_eval.py', '--name', 'x', '--sensor_device', 'device1', '--mode', 'sequential',
                                      '--snr_db', '30', '--sensor_config', str(tmp_path / 'missing.json')])
    with pytest.raises(FileExistsError, match='x.json'):                  # refused before the config is even read
        SE.main()
    monkeypatch.undo()
    assert SE.output_path('run1') == os.path.join(SE.EVAL_DIR, 'run1.json') and SE.EVAL_DIR.endswith('/sensor/eval')


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')                # K = 2 < min_k_for(0.5)
def test_B_is_unchanged_when_off_mask_entries_are_corrupted(small_R, synthetic_prior):
    args = fake_args(mode='snapshot', methods=['B'], K=2, pigdm_steps=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = run(StubSensorDS(small_R, synthetic_prior, mode='snapshot', n=3), args, dict(B=Const(0.0)))
        b = run(StubSensorDS(small_R, synthetic_prior, mode='snapshot', n=3, corrupt=1e3), args, dict(B=Const(0.0)))
    assert set(a['per_cube']) == {'B'} and len(a['per_cube']['B']['crps']) == 3
    assert json.dumps(a['per_cube']['B']) == json.dumps(b['per_cube']['B'])


def test_coverage_exact_flag(small_R, synthetic_prior):
    ds = StubSensorDS(small_R, synthetic_prior, n=2)
    assert run(ds, fake_args(methods=['L'], K=32), {})['coverage_exact'] is True
    assert run(ds, fake_args(methods=['L'], K=8), {})['coverage_exact'] is False
