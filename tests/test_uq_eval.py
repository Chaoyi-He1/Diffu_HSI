"""Phase 2a Task 8: uq_eval scores every method per cube into one record, refuses mismatched checkpoints and refuses
to overwrite; stub models and a stub dataset on CPU."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.diffusion_trainer import DiffusionTrainer
from scripts import uq_eval as UE

ROOT = '/fake/root'


class StubDS(torch.utils.data.Dataset):
    sigma_d = 0.1
    d = 4

    def __init__(self, n=6, H=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.img_list = [os.path.join(ROOT, f'MatFlower60/Train/P02{2 + i % 3}/{i}_1.mat') for i in range(n)]
        self.xh = torch.rand(n, H, H, 64, generator=g) * 1.2 - 0.6
        self.x = (self.xh + 0.1 * torch.randn(n, H, H, 64, generator=g)).clamp(-1, 1)
        self.y = torch.rand(n, H // 4, H // 4, 30, generator=g)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


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


def fake_args(tmp_path, **kw):
    a = SimpleNamespace(d=4, name='t', K=3, sampler='ddim10', subset='gate', device='cpu', batch_size=4, num_workers=0,
                        store_cubes=2, limit_files=0, yswap_n=4, xr_cache='', centre='c', diffusion='d', varhead='v',
                        ensemble=['e1', 'e2'], b=None, centre_halves=None, data_root=ROOT, prior='p')
    a.__dict__.update(kw)
    return a


def test_point_and_sample_and_gaussian_metrics_shapes():
    x = torch.rand(2, 64, 8, 8) * 2 - 1
    pm = UE.point_metrics(x, x)
    assert set(pm) == {'rmse_pct', 'psnr', 'sam_deg', 'hf', 'w1'} and pm['rmse_pct'] == [0.0, 0.0] and pm['hf'] == [0.0, 0.0]
    s = x[None] + 0.02 * torch.randn(5, 2, 64, 8, 8)
    with pytest.warns(UserWarning, match='coverage_samples: K=5 < 19'):          # K = 5 is below min_k_for(0.9)
        sm = UE.sample_metrics(s, x)
    assert len(sm['crps']) == 2 and all(0 < v < 5 for v in sm['crps']) and all(0 <= v <= 1 for v in sm['cov90'])
    assert set(sm) >= {'mean_rmse_pct', 'single_rmse_pct', 'mean_hf', 'single_hf', 'mean_w1', 'single_w1', 'crps', 'cov50', 'cov90',
                       'cov50_gauss', 'cov90_gauss', 'ause', 'spread_ratio'}
    with np.errstate(divide='ignore'):                                           # mu = x: zero error, spread_ratio = inf
        gm = UE.gaussian_metrics(x, torch.full_like(x, 0.02), x)
    assert gm['mean_rmse_pct'] == [0.0, 0.0] and all(v > 0 for v in gm['crps']) and gm['cov90'] == [1.0, 1.0]
    assert set(gm) == {'mean_rmse_pct', 'mean_psnr', 'mean_sam_deg', 'mean_hf', 'mean_w1', 'crps', 'cov50', 'cov90', 'ause', 'spread_ratio'}


def test_draw_samples_seeds_are_reproducible_and_differ_across_k():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None)
    centre = torch.zeros(2, 64, 8, 8)
    cond = {'sensor': torch.zeros(2, 30, 2, 2), 'x0_hat': centre}
    s1 = UE.draw_samples(Noisy(), tr, cond, tuple(centre.shape), 0.1, UE.SAMPLERS['ddim10'], 3, centre)
    s2 = UE.draw_samples(Noisy(), tr, cond, tuple(centre.shape), 0.1, UE.SAMPLERS['ddim10'], 3, centre)
    assert s1.shape == (3, 2, 64, 8, 8) and torch.equal(s1, s2) and not torch.equal(s1[0], s1[1])
    assert s1.abs().max() <= 1.0


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')                # K = 3 < min_k_for(0.9)
def test_run_eval_writes_every_method_and_the_yswap_block(tmp_path):
    ds = StubDS()
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=UE.residual_collate_fn)
    files = ds.img_list[:4]
    ys = (files, *UE.residual_collate_fn([ds[i] for i in range(4)]))
    models = dict(centre=Const(0.0), diffusion=Noisy(), varhead=Const(0.0), ensemble=[Const(0.1), Const(-0.1)], b=Noisy(), halves=None)
    prov = dict(centre={'run': 'c'}, diffusion={'run': 'd'}, varhead=None, ensemble=[], b=None, halves=None, xr_cache=None, prior=None)
    args = fake_args(tmp_path)
    rec = UE.run_eval(models, 0.1, ds, loader, ys, args, prov, torch.device('cpu'), store_dir=str(tmp_path))
    assert rec['n'] == 6 and rec['K'] == 3 and rec['d'] == 4 and rec['sigma_r'] == 0.1 and rec['sigma_d'] == 0.1
    assert rec['coverage_min_k'] == {'0.5': 3, '0.9': 19} and rec['coverage_exact'] is False
    assert set(rec['per_cube']) == {'baseline', 'regressor', 'diffusion', 'varhead', 'ensemble', 'b'}
    for m, d in rec['per_cube'].items():
        for k, v in d.items():
            assert len(v) == 6, (m, k)
    assert rec['per_cube']['regressor']['rmse_pct'] == rec['per_cube']['baseline']['rmse_pct']      # centre = zero residual
    assert rec['means']['diffusion']['crps'] > 0 and 'cov90' in rec['means']['varhead'] and 'crps' in rec['means']['ensemble']
    assert rec['yswap']['K'] == 3 and len(rec['yswap']['true']) == 4 == len(rec['yswap']['swap_full']) and rec['yswap']['files'] == [os.path.relpath(f, ROOT) for f in files]
    assert set(rec['yswap']) == {'K', 'files', 'true', 'swap_full', 'baseline', 'regressor'}
    assert set(rec['provenance']) == {'centre', 'diffusion', 'varhead', 'ensemble', 'b', 'halves', 'xr_cache', 'prior', 'git_head', 'created',
                                      'seeds', 'device', 'batch_size', 'subset', 'limit_files', 'seconds'}
    assert rec['files'] == [os.path.relpath(f, ROOT) for f in ds.img_list] and rec['classes'] == ['P022', 'P023', 'P024'] * 2
    assert rec['provenance']['seeds'] == [0, 1, 2] and rec['stored_cubes'].endswith('__cubes.npz') and os.path.exists(rec['stored_cubes'])
    z = np.load(rec['stored_cubes'])
    assert z['samples'].shape == (3, 2, 64, 8, 8) and z['samples'].dtype == np.float16
    # stored from the y-swap batch (4 cubes), evenly spaced: its first and last cube
    assert len(rec['stored_files']) == min(args.store_cubes, len(files)) and rec['stored_files'] == [rec['yswap']['files'][i] for i in (0, 3)]
    assert z['files'].tolist() == rec['stored_files'] and np.array_equal(z['x'], ys[1][[0, 3]].numpy())
    json.dumps(rec)                                                              # the record is plain JSON
    with pytest.raises(FileExistsError, match='__cubes.npz'):                    # the stored cubes are never overwritten either
        UE.run_eval(models, 0.1, ds, loader, ys, args, prov, torch.device('cpu'), store_dir=str(tmp_path))


@pytest.mark.filterwarnings('ignore:coverage_samples:UserWarning')                # K = 2 < min_k_for(0.5)
def test_run_eval_without_diffusion_or_varhead_is_the_stage_0_dry_run(tmp_path):
    ds = StubDS()
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=UE.residual_collate_fn)
    models = dict(centre=Const(0.0), diffusion=None, varhead=None, ensemble=[Const(0.1), Const(-0.1)], b=Noisy(), halves=None)
    prov = dict(centre={'run': 'c'}, diffusion=None, varhead=None, ensemble=[], b={'run': 'b'}, halves=None, xr_cache=None)
    rec = UE.run_eval(models, None, ds, loader, None, fake_args(tmp_path, diffusion=None, varhead=None, K=2), prov, torch.device('cpu'), store_dir=str(tmp_path))
    assert set(rec['per_cube']) == {'baseline', 'regressor', 'ensemble', 'b'} and rec['yswap'] is None and rec['sigma_r'] is None
    assert rec['stored_cubes'] is None and rec['stored_files'] is None
    assert len(rec['per_cube']['b']['crps']) == 6


def test_run_eval_with_k_at_least_19_marks_coverage_exact(tmp_path):
    ds = StubDS()
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=UE.residual_collate_fn)
    models = dict(centre=Const(0.0), diffusion=None, varhead=None, ensemble=[], b=None, halves=None)
    prov = dict(centre={'run': 'c'}, diffusion=None, varhead=None, ensemble=[], b=None, halves=None, xr_cache=None)
    rec = UE.run_eval(models, None, ds, loader, None, fake_args(tmp_path, diffusion=None, varhead=None, K=19), prov, torch.device('cpu'))
    assert rec['coverage_exact'] is True and set(rec['per_cube']) == {'baseline', 'regressor'} and rec['stored_cubes'] is None


def test_write_record_refuses_overwrite(tmp_path):
    p = str(tmp_path / 'uq' / 'd4' / 'eval' / 'x.json')
    UE.write_record(p, dict(a=1))
    with open(p) as f:
        assert json.load(f) == dict(a=1)
    with pytest.raises(FileExistsError):
        UE.write_record(p, dict(a=2))
    assert UE.output_path(8, 'run1').endswith('/uq/d8/eval/run1.json')


def test_expect_names_every_mismatch():
    UE.expect(dict(mode='regress', d=8), 'p', mode='regress', d=8)
    with pytest.raises(ValueError) as e:
        UE.expect(dict(mode='residual', d=4), 'p', mode='regress', d=8)
    assert 'mode' in str(e.value) and 'd' in str(e.value) and 'p' in str(e.value)


def save_ckpt(tmp_path, name, meta):
    from model.residual_wrapper import build_residual_model
    p = tmp_path / name
    torch.save(dict(model_state_dict=build_residual_model(meta['mode'], 8).state_dict(), meta=meta), p)
    return str(p)


CACHE_META = dict(d=4, sigma_r=0.09, sigma_full=0.1, sigma_ratio=0.1 / 0.09, sigma_flag=True)


def test_check_models_ties_the_diffusion_to_its_centre_and_cache(tmp_path, monkeypatch):
    from data_loader.warmstart_paths import file_sha256
    from data_loader.xr_cache import write_meta
    save = lambda name, meta: save_ckpt(tmp_path, name, meta)
    centre = save('centre.pth', dict(mode='regress', d=4, base_channels=8, sigma_d=0.1, args={}))
    good = dict(mode='residual_reg', d=4, base_channels=8, prediction_type='v', n_timesteps=1000, sigma_d=0.09, sigma_r=0.09,
                xr_cache_sha256='cachesha', centre=dict(run='centre', epoch=1, sha256=file_sha256(centre)), args={})
    diff = save('diff.pth', good)
    cache = str(tmp_path / 'xr_cache')
    write_meta(cache, CACHE_META)
    monkeypatch.setattr(UE, 'meta_sha256', lambda d: 'cachesha')
    models, sigma_r, prov = UE.check_models(fake_args(tmp_path, centre=centre, diffusion=diff, varhead=None, ensemble=[], xr_cache=cache), torch.device('cpu'))
    assert sigma_r == 0.09 and models['diffusion'] is not None and prov['diffusion']['sha256'] == file_sha256(diff)
    assert prov['xr_cache'] == dict(dir=cache, sha256='cachesha', sigma_r=0.09, sigma_full=0.1, sigma_ratio=0.1 / 0.09, sigma_flag=True)
    assert set(prov) == {'centre', 'diffusion', 'varhead', 'ensemble', 'b', 'halves', 'xr_cache', 'prior'}
    bad = save('bad.pth', dict(good, centre=dict(run='other', epoch=1, sha256='nope')))
    with pytest.raises(ValueError, match='centre'):
        UE.check_models(fake_args(tmp_path, centre=centre, diffusion=bad, varhead=None, ensemble=[]), torch.device('cpu'))
    eps = save('eps.pth', dict(good, prediction_type='eps'))                    # would be sampled as v without the check
    with pytest.raises(ValueError, match='prediction_type'):
        UE.check_models(fake_args(tmp_path, centre=centre, diffusion=eps, varhead=None, ensemble=[]), torch.device('cpu'))
    thin = str(tmp_path / 'thin_cache')
    write_meta(thin, {k: v for k, v in CACHE_META.items() if k != 'sigma_full'})
    with pytest.raises(ValueError, match='sigma_full'):
        UE.check_models(fake_args(tmp_path, centre=centre, diffusion=diff, varhead=None, ensemble=[], xr_cache=thin), torch.device('cpu'))
    monkeypatch.setattr(UE, 'meta_sha256', lambda d: 'rebuilt')
    with pytest.raises(ValueError, match='xr_cache'):
        UE.check_models(fake_args(tmp_path, centre=centre, diffusion=diff, varhead=None, ensemble=[], xr_cache=cache), torch.device('cpu'))


def test_check_models_refuses_half_trained_full_models_b_off_schedule_and_split_caches(tmp_path):
    from data_loader.warmstart_paths import file_sha256
    save = lambda name, meta: save_ckpt(tmp_path, name, meta)
    reg = dict(mode='regress', d=4, base_channels=8, sigma_d=0.1, args={})
    centre, half = save('centre.pth', reg), save('half.pth', dict(reg, args=dict(train_half=1)))
    b = dict(mode='residual', d=4, base_channels=8, prediction_type='v', n_timesteps=1000, sigma_d=0.1, args={})
    no_xr = dict(diffusion=None, varhead=None, ensemble=[])
    UE.check_models(fake_args(tmp_path, centre=centre, b=save('b.pth', b), **no_xr), torch.device('cpu'))
    for kw in (dict(centre=half), dict(ensemble=[centre, half]), dict(b=save('b_half.pth', dict(b, args=dict(train_half=2))))):
        with pytest.raises(ValueError, match='train_half'):
            UE.check_models(fake_args(tmp_path, **dict(dict(no_xr, centre=centre), **kw)), torch.device('cpu'))
    with pytest.raises(ValueError, match='n_timesteps'):
        UE.check_models(fake_args(tmp_path, centre=centre, b=save('b500.pth', dict(b, n_timesteps=500)), **no_xr), torch.device('cpu'))
    ctr = dict(run='centre', epoch=1, sha256=file_sha256(centre))
    diff = save('diff.pth', dict(mode='residual_reg', d=4, base_channels=8, prediction_type='v', n_timesteps=1000, sigma_d=0.09,
                                 sigma_r=0.09, xr_cache_sha256='cache_a', centre=ctr, args={}))
    vh = dict(mode='varhead', d=4, base_channels=8, sigma_d=0.09, sigma_r=0.09, xr_cache_sha256='cache_a', centre=ctr, args={})
    UE.check_models(fake_args(tmp_path, centre=centre, diffusion=diff, varhead=save('vh.pth', vh), ensemble=[]), torch.device('cpu'))
    with pytest.raises(ValueError, match='xr_cache_sha256'):                    # no --xr_cache: the two must still share a cache
        UE.check_models(fake_args(tmp_path, centre=centre, diffusion=diff, varhead=save('vh_b.pth', dict(vh, xr_cache_sha256='cache_b')),
                                  ensemble=[]), torch.device('cpu'))


def test_check_models_refuses_a_foreign_prior_and_a_one_member_ensemble(tmp_path):
    from data_loader.warmstart_paths import file_sha256
    prior = tmp_path / 'prior.npz'
    prior.write_bytes(b'not a real prior, only hashed')
    save = lambda name, meta: save_ckpt(tmp_path, name, meta)
    centre = save('centre.pth', dict(mode='regress', d=4, base_channels=8, sigma_d=0.1, prior_sha256=file_sha256(str(prior)), args={}))
    foreign = save('foreign.pth', dict(mode='regress', d=4, base_channels=8, sigma_d=0.1, prior_sha256='q' * 64, args={}))
    no_xr = dict(diffusion=None, varhead=None, ensemble=[])
    _, _, prov = UE.check_models(fake_args(tmp_path, centre=centre, prior=str(prior), **no_xr), torch.device('cpu'))
    assert prov['prior'] == dict(path=str(prior), sha256=file_sha256(str(prior)))
    with pytest.raises(ValueError, match='prior_sha256'):
        UE.check_models(fake_args(tmp_path, centre=foreign, prior=str(prior), **no_xr), torch.device('cpu'))
    with pytest.raises(ValueError, match='prior_sha256'):
        UE.check_models(fake_args(tmp_path, centre=centre, prior=str(prior), **dict(no_xr, ensemble=[centre, foreign])), torch.device('cpu'))
    with pytest.raises(ValueError, match='at least 2'):
        UE.check_models(fake_args(tmp_path, centre=centre, **dict(no_xr, ensemble=[centre])), torch.device('cpu'))
