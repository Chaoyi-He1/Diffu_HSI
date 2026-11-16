"""Phase 3 Task 10: sd_eval scores L / L_demosaic / D / V / C / C1 / B per cube into one record, refuses checkpoints of
another config sha or mode, refuses to overwrite; stub models and a stub sensor dataset on CPU."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_loader.hfd_sensor_sim import HFDSensorSim, sensor_collate_fn
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

    def checkpoint_extra(self):
        return dict(seed_scheme='crc32(relpath)')

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


class Shrink(torch.nn.Module):
    """x0-prediction stub that depends on x_t (a constant x0 would give guidance no gradient path)."""
    def forward(self, x_t, cond, t):
        return 0.5 * x_t


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


def save_ckpt(path, mode, sha=None, prediction_type='v', sensor_device=None):
    meta = dict(mode=mode, base_channels=16, sensor_config_sha=sha or training_sha(CFG), prediction_type=prediction_type)
    if sensor_device is not None:
        meta['sensor_device'] = sensor_device
    torch.save(dict(model_state_dict=build_residual_model(mode, 16).state_dict(), meta=meta), path)
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
    assert abs(s['sigma'] - sigma_from_snr(30.0, 1.0)) < 1e-6 and s['seed_scheme'] == 'crc32(relpath)'
    assert set(rec) == {'name', 'setting', 'subset', 'K', 'pigdm_steps', 'guidance', 'n', 'files', 'classes', 'per_cube',
                        'means', 'coverage_exact', 'provenance'}
    assert set(rec['provenance']) == {'sensor_config_sha', 'config_path', 'checkpoints', 'git_head', 'created', 'seconds',
                                      'seeds', 'device', 'batch_size', 'limit_files', 'b_prediction_type', 'c_sampler',
                                      'b_kernel'}
    assert rec['provenance']['b_kernel'] is None                               # no Bp requested
    assert rec['provenance']['sensor_config_sha'] == training_sha(CFG) and rec['provenance']['seeds'] == [0, 1, 2]
    cs = rec['provenance']['c_sampler']                                   # C's sampler beside B's pigdm_steps / guidance
    assert cs['method'] == 'ddim' and cs['clamp_x0'] == [-8.0, 8.0] and cs['prediction_type'] == 'v'
    assert cs['C']['n_steps'] == 10 and cs['C']['seeds'] == [0, 1, 2]
    assert cs['C1']['n_steps'] == 1 and cs['C1']['seed'] == 0 and 'one DDIM call' in cs['C1']['rule']
    assert SE.c_sampler_provenance(['L', 'D'], 3) is None and set(SE.c_sampler_provenance(['C1'], 3)) == \
        {'method', 'prediction_type', 'n_timesteps', 'clamp_x0', 'C1'}
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


def test_B_accepts_an_uncond_tt_checkpoint(tmp_path):
    p = save_ckpt(tmp_path / 'utt.pth', 'uncond_tt')
    models, prov = SE.check_models(fake_args(methods=['B'], uncond=p), CFG, torch.device('cpu'))
    assert set(models) == {'B'} and prov['uncond']['mode'] == 'uncond_tt' and models['B'].net.time_tokens_when_no_context
    p2 = save_ckpt(tmp_path / 'cf.pth', 'cond_fam')
    with pytest.raises(ValueError, match='cf.pth.*uncond'):                 # other modes are still refused for B
        SE.check_models(fake_args(methods=['B'], uncond=p2), CFG, torch.device('cpu'))


def test_B_accepts_x0_prediction_prior_and_records_it(tmp_path, small_R, synthetic_prior):
    p = save_ckpt(tmp_path / 'ux0.pth', 'uncond', prediction_type='x0')
    args = fake_args(mode='snapshot', methods=['B'], K=2, pigdm_steps=2, uncond=p)
    models, prov = SE.check_models(args, CFG, torch.device('cpu'))
    assert set(models) == {'B'} and prov['uncond']['prediction_type'] == 'x0'
    ds = StubSensorDS(small_R, synthetic_prior, mode='snapshot', n=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = SE.run_eval(dict(B=Shrink()), ds, loader, args, CFG, prov, torch.device('cpu'))
    assert rec['provenance']['b_prediction_type'] == 'x0'
    assert len(rec['per_cube']['B']['crps']) == 3


def test_B_refuses_eps_prediction_prior(tmp_path):
    p = save_ckpt(tmp_path / 'ueps.pth', 'uncond', prediction_type='eps')
    with pytest.raises(ValueError, match='ueps.pth'):
        SE.check_models(fake_args(methods=['B'], uncond=p), CFG, torch.device('cpu'))


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
    # Shrink depends on x_t, so guidance has a gradient path and B's draws depend on y; the on-mask control below proves it
    # (with a constant model the off-mask equality would hold even if B ignored y entirely).
    args = fake_args(mode='snapshot', methods=['B'], K=2, pigdm_steps=2)

    class OnMask(StubSensorDS):
        def __getitem__(self, i):
            d = super().__getitem__(i)
            d['y'] = (d['y'] + 0.5 * d['mask']).astype(np.float32)
            return d

    with np.errstate(divide='ignore', invalid='ignore'):
        a = run(StubSensorDS(small_R, synthetic_prior, mode='snapshot', n=3), args, dict(B=Shrink()))
        b = run(StubSensorDS(small_R, synthetic_prior, mode='snapshot', n=3, corrupt=1e3), args, dict(B=Shrink()))
        c = run(OnMask(small_R, synthetic_prior, mode='snapshot', n=3), args, dict(B=Shrink()))
    assert set(a['per_cube']) == {'B'} and len(a['per_cube']['B']['crps']) == 3
    assert json.dumps(a['per_cube']['B']) == json.dumps(b['per_cube']['B'])
    assert not np.allclose(a['per_cube']['B']['crps'], c['per_cube']['B']['crps'])           # B does read the measured entries


def test_coverage_exact_flag(small_R, synthetic_prior):
    ds = StubSensorDS(small_R, synthetic_prior, n=2)
    assert run(ds, fake_args(methods=['L'], K=32), {})['coverage_exact'] is True
    assert run(ds, fake_args(methods=['L'], K=8), {})['coverage_exact'] is False


def test_records_on_different_subsets_pair_by_file(fake_sensor_root, synthetic_prior, tmp_path):
    np.savez(tmp_path / 'prior.npz', mu=synthetic_prior['mu'], C=synthetic_prior['C'], s=3e-7)

    def sim(files):
        ds = HFDSensorSim(fake_sensor_root, 'train', CFG, device='device1', setting=dict(mode='snapshot', snr_db=30.0),
                          prior_path=str(tmp_path / 'prior.npz'))
        ds.img_list = [ds.img_list[i] for i in files]
        return ds

    big, small = sim([0, 3, 5, 7]), sim([7, 3])                    # the shared files sit at different positions
    args = fake_args(mode='snapshot', methods=['L'], data_root=fake_sensor_root)
    with np.errstate(divide='ignore', invalid='ignore'):
        rb, rs = run(big, args, {}), run(small, args, {})
    assert rb['setting']['seed_scheme'] == rs['setting']['seed_scheme'] == 'crc32(relpath)'
    for f in rs['files']:
        assert rb['per_cube']['L']['mean_rmse_pct'][rb['files'].index(f)] == rs['per_cube']['L']['mean_rmse_pct'][rs['files'].index(f)]


def test_refuses_learned_checkpoint_trained_on_another_device(tmp_path):
    cfg = dict(CFG, device_train='device1')
    for flag, mode, m in (('regress', 'regress_fam', 'D'), ('varhead', 'varhead_fam', 'V'), ('cond', 'cond_fam', 'C')):
        p = save_ckpt(tmp_path / f'{flag}_dev2.pth', mode, sensor_device='device2')
        with pytest.raises(ValueError, match=f"{flag}_dev2.pth.*sensor_device: checkpoint 'device2' vs config device_train 'device1'"):
            SE.check_models(fake_args(methods=[m], **{flag: p}), cfg, torch.device('cpu'))
        ok = save_ckpt(tmp_path / f'{flag}_dev1.pth', mode, sensor_device='device1')
        SE.check_models(fake_args(methods=[m], **{flag: ok}), cfg, torch.device('cpu'))
    legacy = save_ckpt(tmp_path / 'nofield.pth', 'regress_fam')                       # no sensor_device field = device_train
    SE.check_models(fake_args(methods=['D'], regress=legacy), cfg, torch.device('cpu'))
    prior = save_ckpt(tmp_path / 'u_dev2.pth', 'uncond', sensor_device='device2')      # the prior is sensor-free: no check
    SE.check_models(fake_args(methods=['B'], uncond=prior), cfg, torch.device('cpu'))


# ----------------------------------------------------------------------------- B+ (method Bp)
def _kernel(tmp_path, scale=0.01, **meta):
    p = str(tmp_path / 'b_kernel.npz')
    np.savez(p, Sigma_e=scale * np.eye(64), meta=np.array(meta, dtype=object))
    return p


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')
@pytest.mark.parametrize('mode', ['sequential', 'snapshot'])
def test_Bp_rescoring_of_the_same_guided_draws(small_R, synthetic_prior, tmp_path, mode):
    ds = StubSensorDS(small_R, synthetic_prior, mode=mode, n=3)
    kern = _kernel(tmp_path)
    both = run(ds, fake_args(mode=mode, methods=['B', 'Bp'], uncond='u', b_kernel=kern, save_draws=str(tmp_path / 'draws')),
               {'B': Shrink()})
    only_b = run(ds, fake_args(mode=mode, methods=['B'], uncond='u'), {'B': Shrink()})
    assert both['per_cube']['B'] == only_b['per_cube']['B']                    # B is scored on untouched draws
    b, bp = both['means']['B'], both['means']['Bp']
    assert bp['spread_ratio'] > b['spread_ratio'] and bp['single_rmse_pct'] != b['single_rmse_pct']
    assert both['provenance']['b_kernel']['sha256'] == SE.kernel_sha256(0.01 * np.eye(64))
    assert only_b['provenance']['b_kernel'] is None
    d = np.load(str(tmp_path / 'draws' / 't.npz'))
    assert d['draws'].shape == (3, 3, 64, H, H) and d['draws'].dtype == np.float16 and len(d['files']) == 3
    alone = run(ds, fake_args(mode=mode, methods=['Bp'], uncond='u', b_kernel=kern), {'B': Shrink()})
    assert alone['per_cube']['Bp'] == both['per_cube']['Bp'] and 'B' not in alone['per_cube']


def test_Bp_needs_the_kernel_and_the_kernel_must_fit_the_prior(small_R, synthetic_prior, tmp_path):
    with pytest.raises(ValueError, match='b_kernel'):
        SE.resolve_methods(fake_args(methods=['Bp'], uncond='u'))
    assert SE.resolve_methods(fake_args(uncond='u')) == ['L', 'B']
    assert SE.resolve_methods(fake_args(uncond='u', b_kernel='k')) == ['L', 'B', 'Bp']
    ds = StubSensorDS(small_R, synthetic_prior, n=2)
    args = fake_args(methods=['Bp'], uncond='u', b_kernel=_kernel(tmp_path, uncond_sha256='a' * 64))
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
    with pytest.raises(ValueError, match='another prior checkpoint'):
        SE.run_eval({'B': Shrink()}, ds, loader, args, CFG, {'uncond': {'sha256': 'b' * 64, 'prediction_type': 'v'}},
                    torch.device('cpu'))


# ----------------------------------------------------------------------------- Phase 3b scenarios (--scenario)
from data_loader import scenarios as SCN  # noqa: E402
from data_loader.mmse import mmse_maps_general  # noqa: E402
from data_loader.warmstart_paths import file_sha256  # noqa: E402
import model.guidance as GD  # noqa: E402

SCN_CONSTS = dict(mean_y=3.0, y_max=9.0, quantiles={'0.9': 5.0, '0.7': 4.0})
OLD_SETTING_KEYS = {'sensor_device', 'mode', 'snr_db', 'sigma', 'R_sha256', 'seed_scheme'}


class StubScenarioDS(StubSensorDS):
    """HFDSensorSim's scenario items for one SCENARIOS entry (R2 = the flipped fixture R)."""

    def __init__(self, R, prior, name, consts=None, n=4):
        spec = SE.SCENARIOS[name]
        super().__init__(R, prior, snr_db=spec['snr_db'], n=n)
        self.scen, self.consts = spec['scenario'], consts
        self.R = SCN.scenario_R(R, np.flipud(R).copy(), self.scen)
        self.R_sha256 = 'scenario-r'

    def __getitem__(self, i):
        x01 = self.cubes[i]
        sigma = sigma_from_snr(self.snr_db, CFG['P_y'])
        m = SCN.measure_scenario(x01, self.R, sigma, self.scen, np.random.default_rng(i), self.consts)
        xh, s = mmse_maps_general(m['y'], m['mask'], m['noise_var'], self.R, self.prior, self.A)
        out = dict(x=((2.0 * x01 - 1.0) @ self.A.T).astype(np.float32), y=m['y'], x0_hat=xh,
                   s_post=np.maximum(s, SIGMA_D).astype(np.float32), s_raw=s, mask=m['mask'], sigma=np.float32(sigma),
                   snr_db=np.float32(self.snr_db), snapshot=np.float32(0.0), noise_var=m['noise_var'])
        if SCN.has_naive(self.scen):
            xn, sn = mmse_maps_general(m['y_naive'], m['mask_naive'], m['noise_var_naive'], self.R, self.prior, self.A)
            out.update(y_naive=m['y_naive'], mask_naive=m['mask_naive'], noise_var_naive=m['noise_var_naive'],
                       x0_hat_naive=xn, s_post_naive=np.maximum(sn, SIGMA_D).astype(np.float32), s_raw_naive=sn)
        if SCN.has_pixels(self.scen):
            out['measured'] = m['measured'][..., None].astype(np.float32)
        return out


class SpyMeasurement(GD.LinearMeasurement):
    made = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        SpyMeasurement.made.append(self)


def _spy(monkeypatch):
    SpyMeasurement.made = []
    calls = []
    real = SE.posterior_draws

    def pd(draws, H, b, y, sigma2, Sigma_e, mask=None, clamp=(-1.0, 1.0)):
        calls.append(dict(sigma2=sigma2, mask=mask))
        return real(draws, H, b, y, sigma2, Sigma_e, mask=mask, clamp=clamp)
    monkeypatch.setattr(SE, 'LinearMeasurement', SpyMeasurement)
    monkeypatch.setattr(SE, 'posterior_draws', pd)
    return calls


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')
def test_pixel_scenario_record_carries_L_interp_and_e2(small_R, synthetic_prior, tmp_path, monkeypatch):
    ds = StubScenarioDS(small_R, synthetic_prior, 'stride_4_30')
    monkeypatch.setattr(SE, 'calibration_cubes01', lambda data_root, n=SE.N_E2_CUBES: ds.cubes)
    calls = _spy(monkeypatch)
    args = fake_args(scenario='stride_4_30', scenario_consts=None, uncond='u', b_kernel=_kernel(tmp_path))
    assert SE.resolve_methods(args) == ['L', 'L_interp', 'B', 'Bp']
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = run(ds, args, {'B': Shrink()})
    assert list(rec['per_cube']) == ['L', 'L_interp', 'B', 'Bp'] and len(rec['per_cube']['L_interp']['crps']) == 4
    s = rec['setting']
    assert set(s) == OLD_SETTING_KEYS | {'scenario', 'scenario_sha', 'e2_interp'}
    assert s['scenario'] == SE.SCENARIOS['stride_4_30']['scenario'] and s['scenario_sha'] == SCN.scenario_sha(s['scenario'])
    assert s['R_sha256'] == 'scenario-r'
    np.testing.assert_allclose(s['e2_interp'], SCN.interp_error_e2(ds.cubes, ds.R, {'kind': 'stride', 'd': 4}, 0))
    assert rec['provenance']['scenario_consts'] is None
    assert set(rec) == {'name', 'setting', 'subset', 'K', 'pigdm_steps', 'guidance', 'n', 'files', 'classes', 'per_cube',
                        'means', 'coverage_exact', 'provenance'}
    # the guidance and the B+ conditioning see the stride mask (gaussian noise: no per-element variance)
    assert SpyMeasurement.made and all(m.general for m in SpyMeasurement.made)
    assert all(not torch.is_tensor(c['sigma2']) and c['mask'] is not None for c in calls)
    # L_interp by hand for cube 0: the readings interpolated over the image, then the sequential MMSE with sigma^2 + e2
    it = ds[0]
    xh, sd = SCN.interp_then_mmse(it['y'], it['measured'][..., 0] > 0.5, ds.R, ds.prior, float(it['sigma']), ds.A,
                                  np.asarray(s['e2_interp']))
    ref = SE.gaussian_metrics(torch.from_numpy(xh.transpose(2, 0, 1))[None], torch.from_numpy(sd.transpose(2, 0, 1))[None],
                              torch.from_numpy(it['x'].transpose(2, 0, 1))[None])
    assert np.isclose(rec['per_cube']['L_interp']['crps'][0], ref['crps'][0], rtol=1e-5)
    json.dumps(rec)


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')
def test_poisson_scenario_carries_the_naive_methods_and_aware_guidance(small_R, synthetic_prior, tmp_path, monkeypatch):
    cpath = tmp_path / 'scenario_consts.json'
    cpath.write_text(json.dumps(SCN_CONSTS))
    ds = StubScenarioDS(small_R, synthetic_prior, 'poisson_30', consts=SCN_CONSTS)
    calls = _spy(monkeypatch)
    args = fake_args(scenario='poisson_30', scenario_consts=str(cpath), regress='r', uncond='u', b_kernel=_kernel(tmp_path))
    assert SE.resolve_methods(args) == ['L', 'L_naive', 'D', 'D_naive', 'B', 'Bp']
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = run(ds, args, {'D': Const(0.0), 'B': Shrink()})
    assert list(rec['per_cube']) == ['L', 'L_naive', 'D', 'D_naive', 'B', 'Bp']
    assert rec['per_cube']['L'] != rec['per_cube']['L_naive'] and rec['per_cube']['D'] != rec['per_cube']['D_naive']
    b = sensor_collate_fn([ds[i] for i in range(len(ds))])
    assert rec['per_cube']['D_naive']['rmse_pct'] == SE.point_metrics(b['x0_hat_naive'].clamp(-1, 1), b['x'])['rmse_pct']
    assert rec['provenance']['scenario_consts'] == dict(path=os.path.abspath(str(cpath)), sha256=file_sha256(str(cpath)))
    assert set(rec['setting']) == OLD_SETTING_KEYS | {'scenario', 'scenario_sha', 'e2_interp'} and rec['setting']['e2_interp'] is None
    assert SpyMeasurement.made and all(m.general and m.noise_var is not None for m in SpyMeasurement.made)
    assert torch.equal(SpyMeasurement.made[0].noise_var, b['noise_var'][:2])
    assert all(torch.is_tensor(c['sigma2']) and c['mask'] is None for c in calls)    # all-ones mask: passed as None


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')
def test_gaussian_scenario_without_pixels_uses_the_phase3_measurement(small_R, synthetic_prior, tmp_path, monkeypatch):
    ds = StubScenarioDS(small_R, synthetic_prior, 'states_5_20', n=2)
    calls = _spy(monkeypatch)
    args = fake_args(scenario='states_5_20', scenario_consts=None, uncond='u', b_kernel=_kernel(tmp_path))
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = run(ds, args, {'B': Shrink()})
    assert list(rec['per_cube']) == ['L', 'B', 'Bp'] and ds.R.shape == (31, 5)
    assert all(not m.general and not m.snapshot and m.H.shape[0] == 5 for m in SpyMeasurement.made)
    assert all(not torch.is_tensor(c['sigma2']) and c['mask'] is None for c in calls)


def test_scenario_method_refusals():
    with pytest.raises(ValueError, match='L_naive'):
        SE.resolve_methods(fake_args(methods=['L', 'L_naive']))                         # no scenario
    with pytest.raises(ValueError, match='D_naive'):
        SE.resolve_methods(fake_args(methods=['D_naive'], regress='r', scenario='stride_4_30'))   # gaussian noise
    with pytest.raises(ValueError, match='L_interp'):
        SE.resolve_methods(fake_args(methods=['L_interp'], scenario='poisson_30'))      # no pixel pattern
    with pytest.raises(ValueError, match='L_interp'):
        SE.resolve_methods(fake_args(methods=['L_interp']))
    assert SE.resolve_methods(fake_args(scenario='sat_70')) == ['L', 'L_naive']         # D_naive needs --regress
    assert SE.resolve_methods(fake_args(scenario='dead_90', methods=['L_interp'])) == ['L_interp']
    assert SE.resolve_methods(fake_args(regress='r')) == ['L', 'D']                    # no scenario: as before
    with pytest.raises(ValueError, match='unknown scenario'):
        SE.resolve_methods(fake_args(scenario='nope'))


def test_scenario_arguments(tmp_path):
    ap = SE.get_parser()
    a = ap.parse_args(['--name', 'x', '--scenario', 'blend_050'])
    assert a.snr_db is None and SE.apply_scenario_args(a) == SE.SCENARIOS['blend_050']['scenario']
    assert a.snr_db == 30.0 and a.sensor_device == 'device1' and a.scenario_consts is None
    a = ap.parse_args(['--name', 'x', '--scenario', 'states_5_20', '--snr_db', '20'])
    assert SE.apply_scenario_args(a)['states'] == 5
    with pytest.raises(ValueError, match='snr_db'):
        SE.apply_scenario_args(ap.parse_args(['--name', 'x', '--scenario', 'states_5_20', '--snr_db', '40']))
    with pytest.raises(ValueError, match='sensor_device'):
        SE.apply_scenario_args(ap.parse_args(['--name', 'x', '--scenario', 'blend_050', '--sensor_device', 'device2']))
    with pytest.raises(ValueError, match='sequential'):
        SE.apply_scenario_args(ap.parse_args(['--name', 'x', '--scenario', 'blend_050', '--mode', 'snapshot']))
    with pytest.raises(ValueError, match='snr_db'):
        SE.apply_scenario_args(ap.parse_args(['--name', 'x']))                         # no scenario: --snr_db required
    a = ap.parse_args(['--name', 'x', '--snr_db', '30'])
    assert SE.apply_scenario_args(a) is None and a.snr_db == 30.0
    with pytest.raises(FileNotFoundError, match='scenario_consts'):
        SE.apply_scenario_args(ap.parse_args(['--name', 'x', '--scenario', 'quant_4',
                                              '--scenario_consts', str(tmp_path / 'missing.json')]))
    c = tmp_path / 'c.json'
    c.write_text(json.dumps(SCN_CONSTS))
    a = ap.parse_args(['--name', 'x', '--scenario', 'quant_4', '--scenario_consts', str(c)])
    SE.apply_scenario_args(a)
    assert a.scenario_consts == str(c) and a.snr_db == 40.0
    a = ap.parse_args(['--name', 'x', '--scenario', 'sat_90'])
    if not os.path.isfile(SE.SCENARIO_CONSTS):
        with pytest.raises(FileNotFoundError):
            SE.apply_scenario_args(a)
    assert SE.SCENARIO_CONSTS.endswith('/sensor/scenario_consts.json')


def test_no_scenario_record_has_no_scenario_keys(small_R, synthetic_prior):
    rec = run(StubSensorDS(small_R, synthetic_prior, n=2), fake_args(methods=['L'], K=32), {})
    assert set(rec['setting']) == OLD_SETTING_KEYS and 'scenario_consts' not in rec['provenance']


@pytest.mark.dataset
def test_calibration_cubes_for_e2(data_root):
    cubes = SE.calibration_cubes01(data_root, n=2)
    assert len(cubes) == 2 and cubes[0].shape == (64, 64, 31) and 0 <= cubes[0].min() and cubes[0].max() <= 1
