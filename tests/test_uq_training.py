"""Phase 2a Task 4: the trainer accepts the two new modes and the half/cache flags, refuses the contradictory
combinations, records sigma_r / cache sha / centre / train_half in the checkpoint meta, refuses to resume against a
rebuilt cache, and trains both modes on a stub cache dataset (CPU)."""
import json
import os
from types import SimpleNamespace

import pytest
import torch

import main_2d_residual_fsdp as tr
from data_loader.warmstart_paths import file_sha256
from model.diffusion_trainer import DiffusionTrainer
from model.residual_wrapper import build_residual_model
from model.varhead_trainer import VarheadTrainer
from scripts.uq.split_halves import write_split

CPU = torch.device('cpu')


class StubXRSet(torch.utils.data.Dataset):
    """(x64, y_d, x_r) HWC float32 with a known residual r = (x - x_r) / sigma_r = 0.5 * x_r, plus what the trainer
    reads from an XRCachedData: sigma_d (= sigma_r), prior, img_list, checkpoint_extra()."""
    sigma_r = sigma_d = 0.05
    prior = {'s': 3e-7}

    def __init__(self, n=8, H=16, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.xr = torch.rand(n, H, H, 64, generator=g) * 1.6 - 0.8
        self.x = (self.xr + self.sigma_r * 0.5 * self.xr).clamp(-1, 1)
        self.y = torch.rand(n, H // d, H // d, 30, generator=g)
        self.img_list = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}_1.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xr[i].numpy()

    def checkpoint_extra(self):
        return dict(sigma_r=self.sigma_r, xr_cache_sha256='stubsha', centre=dict(run='stub', epoch=1, sha256='c', train_half=0), split_sha256='s')


def split_file(tmp_path, name='split.json', seed=0):
    """A tiny cross-fitting split on disk (its sha256 is what a half regressor's meta records)."""
    path = str(tmp_path / name)
    write_split(path, dict(seed=seed, half1=['MatFlower60/Train/P022/f0_1.mat'], half2=['MatFlower60/Train/P023/f1_1.mat']))
    return path


def args_for(tmp_path, mode, *extra):
    base = ['--mode', mode, '--device', 'cpu', '--base_channels', '8', '--batch_size', '2', '--num_workers', '0',
            '--num_epochs', '2', '--save_dir', str(tmp_path / 'run'), '--no_bf16', '--save_every', '1', '--val_every', '0']
    return tr.get_parser().parse_args(base + list(extra))


@pytest.mark.parametrize('mode, extra, ok', [
    ('residual_reg', ('--xr_cache', '/c'), True),
    ('residual_reg', (), False),                                               # cache required
    ('residual_reg', ('--xr_cache', '/c', '--prediction_type', 'direct'), False),
    ('residual_reg', ('--xr_cache', '/c', '--loss_type', 'nll'), False),       # nll is varhead's
    ('varhead', ('--xr_cache', '/c', '--prediction_type', 'direct', '--loss_type', 'nll'), True),
    ('varhead', ('--xr_cache', '/c', '--prediction_type', 'direct'), False),   # needs nll
    ('varhead', ('--xr_cache', '/c', '--prediction_type', 'v', '--loss_type', 'nll'), False),
    ('varhead', ('--prediction_type', 'direct', '--loss_type', 'nll'), False),
    ('regress', ('--prediction_type', 'direct', '--train_half', '1'), True),
    ('regress', ('--prediction_type', 'direct', '--xr_cache', '/c'), False),   # cache only for XR modes
    ('residual', ('--train_half', '1'), False),                                # halves are for regressors
    ('residual', (), True),
])
def test_check_mode_args(tmp_path, mode, extra, ok):
    a = args_for(tmp_path, mode, *extra)
    if ok:
        tr.check_mode_args(a)
    else:
        with pytest.raises(ValueError):
            tr.check_mode_args(a)


def test_make_trainer_for_the_new_modes(tmp_path):
    a = args_for(tmp_path, 'varhead', '--xr_cache', '/c', '--prediction_type', 'direct', '--loss_type', 'nll')
    assert isinstance(tr.make_trainer(a, CPU), VarheadTrainer)
    a = args_for(tmp_path, 'residual_reg', '--xr_cache', '/c')
    t = tr.make_trainer(a, CPU)
    assert isinstance(t, DiffusionTrainer) and t.prediction_type == 'v'


def test_checkpoint_meta_carries_train_half_and_the_cache_extra(tmp_path):
    sp = split_file(tmp_path)
    a = args_for(tmp_path, 'regress', '--prediction_type', 'direct', '--train_half', '2', '--split_json', sp)
    m = tr.checkpoint_meta(a, 0.05, 3e-7, 'psha')
    assert m['train_half'] == 2 and m['split_sha256'] == file_sha256(sp) and 'sigma_r' not in m
    with pytest.raises(ValueError, match='split_sha256'):                     # never silently overwritten by an extra
        tr.checkpoint_meta(a, 0.05, 3e-7, 'psha', extra=dict(split_sha256='other'))
    m = tr.checkpoint_meta(args_for(tmp_path, 'regress', '--prediction_type', 'direct'), 0.05, 3e-7, 'psha')
    assert m['train_half'] == 0 and 'split_sha256' not in m                   # a full-data regressor names no split
    a = args_for(tmp_path, 'residual_reg', '--xr_cache', '/c')
    m = tr.checkpoint_meta(a, 0.09, 3e-7, 'psha', extra=StubXRSet().checkpoint_extra())
    assert m['sigma_d'] == 0.09 and m['sigma_r'] == 0.05 and m['xr_cache_sha256'] == 'stubsha' and m['centre']['run'] == 'stub' and m['train_half'] == 0
    assert m['split_sha256'] == 's'                                           # the cache's split


def test_resume_refuses_a_rebuilt_cache_and_another_half(tmp_path):
    a = args_for(tmp_path, 'residual_reg', '--xr_cache', '/c')
    ck = dict(meta=tr.checkpoint_meta(a, 0.05, 3e-7, 'psha', extra=StubXRSet().checkpoint_extra()))
    assert tr.check_resume_meta(ck, a, 0.05, 'psha', cache_sha='stubsha', warn=False)
    with pytest.raises(ValueError, match='xr_cache_sha256'):
        tr.check_resume_meta(ck, a, 0.05, 'psha', cache_sha='othersha', warn=False)
    sp = split_file(tmp_path)
    a1 = args_for(tmp_path, 'regress', '--prediction_type', 'direct', '--train_half', '1', '--split_json', sp)
    ck1 = dict(meta=tr.checkpoint_meta(a1, 0.05, 3e-7, 'psha'))
    assert tr.check_resume_meta(ck1, a1, 0.05, 'psha', warn=False)
    a2 = args_for(tmp_path, 'regress', '--prediction_type', 'direct', '--train_half', '2', '--split_json', sp)
    with pytest.raises(ValueError, match='train_half'):
        tr.check_resume_meta(ck1, a2, 0.05, 'psha', warn=False)
    a1_other = args_for(tmp_path, 'regress', '--prediction_type', 'direct', '--train_half', '1',
                        '--split_json', split_file(tmp_path, 'other_split.json', seed=1))
    with pytest.raises(ValueError, match='split_sha256'):
        tr.check_resume_meta(ck1, a1_other, 0.05, 'psha', warn=False)


class StubBase:
    """What make_datasets reads from an HFDResidualData: img_list and sigma_d."""
    def __init__(self, files):
        self.img_list, self.sigma_d = list(files), 0.09


def test_make_datasets_takes_the_minimal_namespace_of_the_monitoring_scripts(monkeypatch):
    """scripts/regress_monitoring_energy.py and scripts/diagnose_spatial_split.py pass only data_root, d, prior and
    limit_files: no half filter, no cache wrap."""
    gate = json.load(open(tr.SUBSET))['files']
    train_files = [f'/root/MatFlower60/Train/P0{10 + i}/f{i}_1.mat' for i in range(4)]
    val_files = [os.path.join('/root', f) for f in gate] + ['/root/MatFlower60/Train/P023/other.mat'] * 3
    made = {}

    def fake(root, split, **kw):
        made[split] = StubBase(train_files if split == 'train' else val_files)
        return made[split]
    monkeypatch.setattr(tr, 'HFDResidualData', fake)
    train, val = tr.make_datasets(SimpleNamespace(data_root='/root', d=4, prior='/nonexistent', limit_files=0))
    assert train is made['train'] and val is made['test']
    assert train.img_list == train_files and len(val.img_list) == 3


def parts_for(args, ds):
    loader = tr.make_loader(ds, args, batch_size=args.batch_size, shuffle=True, rank=0, world=1, device=CPU)
    model = build_residual_model(args.mode, args.base_channels)
    trainer = tr.make_trainer(args, CPU)
    opt = tr.make_optimizer(model, args)
    return ds, loader, model, trainer, opt, tr.make_scheduler(opt, args)


@pytest.mark.parametrize('mode, extra', [('residual_reg', ('--xr_cache', '/c')),
                                         ('varhead', ('--xr_cache', '/c', '--prediction_type', 'direct', '--loss_type', 'nll'))])
def test_both_modes_train_on_the_stub_cache_and_write_the_meta(tmp_path, mode, extra):
    a = args_for(tmp_path, mode, *extra)
    os.makedirs(a.save_dir)
    ds = StubXRSet()
    tr.run_epochs(a, parts_for(a, ds), tr.NO_DIST, CPU)
    log = [json.loads(l) for l in open(os.path.join(a.save_dir, 'train_log.jsonl'))]
    assert [r['epoch'] for r in log] == [1, 2] and all(torch.isfinite(torch.tensor(r['loss'])) for r in log)
    ck = torch.load(os.path.join(a.save_dir, 'checkpoint_epoch_2.pth'), map_location='cpu', weights_only=False)
    m = ck['meta']
    assert m['mode'] == mode and m['sigma_d'] == 0.05 and m['sigma_r'] == 0.05 and m['xr_cache_sha256'] == 'stubsha' and m['centre']['run'] == 'stub'
    if mode == 'varhead':
        assert log[1]['loss'] < log[0]['loss']
