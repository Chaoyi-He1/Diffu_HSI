"""Phase 3 Task 7: the trainer runs the four sensor modes (uncond / cond_fam / regress_fam / varhead_fam) on dict
batches, refuses the contradictory flag combinations, redraws the training noise per epoch (non-persistent workers),
keeps the monitoring noise fixed, records the sensor config sha in the meta and refuses to resume against another
config; the legacy tuple path still trains (CPU, stub datasets)."""
import json
import math
import os

import numpy as np
import pytest
import torch

import main_2d_residual_fsdp as tr
from data_loader.hfd_residual import residual_collate_fn
from data_loader.hfd_sensor_sim import sensor_collate_fn
from model.residual_wrapper import build_residual_model

CPU = torch.device('cpu')


class StubSensorSet(torch.utils.data.Dataset):
    """Items are the HFDSensorSim dict (HWC float32 maps + scalars); the noise in y / x0_hat depends on (epoch, idx),
    so set_epoch re-seeds it. Carries what the trainer reads: sigma_d, prior, img_list, collate_fn, checkpoint_extra."""
    sigma_d = 0.02
    prior = {'s': 3e-7}
    collate_fn = staticmethod(sensor_collate_fn)

    def __init__(self, n=8, H=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = (torch.rand(n, H, H, 64, generator=g) * 1.6 - 0.8).numpy()
        self.H, self.epoch = H, 0
        self.img_list = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}_1.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def set_epoch(self, e):
        self.epoch = int(e)

    def __getitem__(self, i):
        rng = np.random.default_rng([7, self.epoch, i])
        H = self.H
        snapshot = bool(i % 2)
        if snapshot:
            mask = np.zeros((H, H, 30), np.float32)
            mask[np.arange(H)[:, None], np.arange(H)[None, :], rng.integers(0, 30, (H, H))] = 1.0
        else:
            mask = np.ones((H, H, 30), np.float32)
        y = rng.random((H, H, 30)).astype(np.float32) * mask
        s_raw = (0.01 + 0.05 * rng.random((H, H, 64))).astype(np.float32)
        x0_hat = np.clip(self.x[i] + 0.05 * rng.standard_normal((H, H, 64)), -1, 1).astype(np.float32)
        return dict(x=self.x[i], y=y, x0_hat=x0_hat, s_post=np.maximum(s_raw, self.sigma_d).astype(np.float32), s_raw=s_raw,
                    mask=mask, sigma=np.float32(0.01), snr_db=np.float32(30.0), snapshot=np.float32(snapshot))

    def checkpoint_extra(self):
        return dict(sensor_config_sha='cfgsha', s_floor01=0.01, P_y=50.0, snr_floor_db=60.0, family={},
                    device_train='device1', sensor_device='device1', R_sha256='r')


class StubTupleSet(torch.utils.data.Dataset):
    """The Phase 1 tuple dataset (tests/test_train_residual.py::TinyResidualSet)."""
    sigma_d = 0.05
    prior = {'s': 3e-7}

    def __init__(self, n=8, H=16, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, H, H, 64, generator=g) * 2 - 1
        self.y = torch.rand(n, H // d, H // d, 30, generator=g)
        self.xh = (self.x + 0.05 * torch.randn(n, H, H, 64, generator=g)).clamp(-1, 1)
        self.img_list = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


MODE_FLAGS = {'uncond': ('--prediction_type', 'v', '--loss_type', 'l1'),
              'cond_fam': ('--prediction_type', 'v', '--loss_type', 'l1'),
              'regress_fam': ('--prediction_type', 'direct', '--loss_type', 'l1'),
              'varhead_fam': ('--prediction_type', 'direct', '--loss_type', 'nll')}


def args_for(tmp_path, mode, *extra):
    base = ['--mode', mode, '--device', 'cpu', '--base_channels', '16', '--batch_size', '4', '--num_workers', '0',
            '--num_epochs', '2', '--save_dir', str(tmp_path / 'run'), '--no_bf16', '--save_every', '1', '--val_every', '1']
    return tr.get_parser().parse_args(base + list(extra))


def sensor_args(tmp_path, mode, *extra):
    return args_for(tmp_path, mode, '--ds', '1', '--sensor_config', '/cfg.json', *MODE_FLAGS[mode], *extra)


def parts_for(args, ds):
    loader = tr.make_loader(ds, args, batch_size=args.batch_size, shuffle=True, rank=0, world=1, device=CPU)
    model = build_residual_model(args.mode, args.base_channels)
    trainer = tr.make_trainer(args, CPU)
    opt = tr.make_optimizer(model, args)
    return ds, loader, model, trainer, opt, tr.make_scheduler(opt, args)


def test_check_mode_args_refusals(tmp_path):
    for mode in MODE_FLAGS:                                                     # the four valid sensor launches pass
        tr.check_mode_args(sensor_args(tmp_path, mode))
    bad = [args_for(tmp_path, 'uncond', '--ds', '1'),                                              # no --sensor_config
           args_for(tmp_path, 'cond_fam', '--ds', '4', '--sensor_config', '/cfg.json'),           # sensor modes need d = 1
           args_for(tmp_path, 'regress_fam', '--ds', '1', '--sensor_config', '/cfg.json'),        # needs direct
           args_for(tmp_path, 'varhead_fam', '--ds', '1', '--sensor_config', '/cfg.json', '--prediction_type', 'direct'),  # needs nll
           args_for(tmp_path, 'residual', '--sensor_config', '/cfg.json'),                        # config is sensor-only
           args_for(tmp_path, 'cond_fam', '--ds', '1', '--sensor_config', '/cfg.json', '--prediction_type', 'direct'),
           args_for(tmp_path, 'cond_fam', '--ds', '1', '--sensor_config', '/cfg.json', '--loss_type', 'nll')]
    for a in bad:
        with pytest.raises(ValueError):
            tr.check_mode_args(a)
    tr.check_mode_args(args_for(tmp_path, 'residual'))                          # legacy launches unchanged
    tr.check_mode_args(args_for(tmp_path, 'regress', '--prediction_type', 'direct'))


def test_make_loader_uses_the_dataset_collate_and_no_persistent_workers_with_set_epoch(tmp_path):
    a = sensor_args(tmp_path, 'cond_fam', '--num_workers', '1')
    loader = tr.make_loader(StubSensorSet(), a, 4, True, 0, 1, CPU)
    assert loader.collate_fn is sensor_collate_fn and loader.persistent_workers is False
    legacy = tr.make_loader(StubTupleSet(), args_for(tmp_path, 'residual', '--num_workers', '1'), 4, True, 0, 1, CPU)
    assert legacy.collate_fn is residual_collate_fn and legacy.persistent_workers is True


@pytest.mark.parametrize('mode,pt,loss', [('uncond', 'v', 'l1'), ('cond_fam', 'v', 'l1'), ('regress_fam', 'direct', 'l1'),
                                          ('varhead_fam', 'direct', 'nll')])
def test_two_epochs_train_on_cpu_and_log_finite_losses(tmp_path, mode, pt, loss):
    a = sensor_args(tmp_path, mode)
    assert (a.prediction_type, a.loss_type) == (pt, loss)
    os.makedirs(a.save_dir)
    ds = StubSensorSet()
    tr.run_epochs(a, parts_for(a, ds), tr.NO_DIST, CPU)
    log = [json.loads(l) for l in open(os.path.join(a.save_dir, 'train_log.jsonl'))]
    assert [r['epoch'] for r in log] == [1, 2] and all(math.isfinite(r['loss']) for r in log)
    assert all(r['steps'] == 2 and r['skipped_nonfinite'] == 0 for r in log)
    assert open(os.path.join(a.save_dir, 'DONE')).read().strip() == '2'
    m = torch.load(os.path.join(a.save_dir, 'checkpoint_epoch_2.pth'), map_location='cpu', weights_only=False)['meta']
    assert m['mode'] == mode and m['d'] == 1 and m['sensor_config_sha'] == 'cfgsha' and m['sigma_d'] == 0.02
    assert m['P_y'] == 50.0 and m['sensor_device'] == 'device1' and m['R_sha256'] == 'r'
    assert ds.epoch == 1                                                        # run_epochs called set_epoch(1) last


def test_set_epoch_changes_training_noise_but_monitoring_is_fixed(tmp_path):
    a = sensor_args(tmp_path, 'cond_fam')
    train, val = StubSensorSet(), StubSensorSet()

    def first_y(ds, shuffle, epoch=None):
        loader = tr.make_loader(ds, a, 4, shuffle, 0, 1, CPU)
        if epoch is not None:
            loader.sampler.set_epoch(0)                                         # same order: only the noise may differ
            ds.set_epoch(epoch)
        return next(iter(loader))['y']
    assert not torch.equal(first_y(train, True, 0), first_y(train, True, 1))
    assert torch.equal(first_y(val, False), first_y(val, False))
    assert val.epoch == 0


def test_resume_refused_when_sensor_config_sha_differs(tmp_path):
    a = sensor_args(tmp_path, 'cond_fam')
    ds = StubSensorSet()
    ck = dict(meta=tr.checkpoint_meta(a, ds.sigma_d, 3e-7, 'psha', extra=ds.checkpoint_extra()))
    assert tr.check_resume_meta(ck, a, 0.02, 'psha', warn=False, sensor_sha='cfgsha')
    ck['meta']['sensor_config_sha'] = 'other'
    with pytest.raises(ValueError, match='sensor_config_sha'):
        tr.check_resume_meta(ck, a, 0.02, 'psha', warn=False, sensor_sha='cfgsha')


def test_build_everything_passes_the_sensor_sha_on_resume(tmp_path, monkeypatch):
    a = sensor_args(tmp_path, 'cond_fam')
    ds, _, model, _, opt, sched = parts_for(a, StubSensorSet())
    path = str(tmp_path / 'run' / 'checkpoint_epoch_1.pth')
    tr.save_checkpoint(model, opt, sched, 0, 0.5, tr.checkpoint_meta(a, ds.sigma_d, 3e-7, 'p' * 64, extra=ds.checkpoint_extra()),
                       path, 0.0, tr.NO_DIST)
    monkeypatch.setattr(tr, 'make_datasets', lambda args: (StubSensorSet(), StubSensorSet()))
    monkeypatch.setattr(tr, 'file_sha256', lambda p: 'p' * 64)
    a.resume = path
    assert tr.build_everything(a, tr.NO_DIST, CPU).start_epoch == 1

    class OtherConfig(StubSensorSet):
        def checkpoint_extra(self):
            return dict(super().checkpoint_extra(), sensor_config_sha='other')
    monkeypatch.setattr(tr, 'make_datasets', lambda args: (OtherConfig(), OtherConfig()))
    with pytest.raises(ValueError, match='sensor_config_sha'):
        tr.build_everything(a, tr.NO_DIST, CPU)


def test_legacy_residual_mode_still_trains(tmp_path):
    a = args_for(tmp_path, 'residual', '--num_epochs', '1', '--batch_size', '2')
    os.makedirs(a.save_dir)
    tr.run_epochs(a, parts_for(a, StubTupleSet()), tr.NO_DIST, CPU)
    log = [json.loads(l) for l in open(os.path.join(a.save_dir, 'train_log.jsonl'))]
    assert len(log) == 1 and math.isfinite(log[0]['loss']) and log[0]['steps'] == 4
    m = torch.load(os.path.join(a.save_dir, 'checkpoint_epoch_1.pth'), map_location='cpu', weights_only=False)['meta']
    assert m['mode'] == 'residual' and m['sigma_d'] == 0.05 and 'sensor_config_sha' not in m


def test_resume_refused_when_sensor_device_differs(tmp_path):
    a = sensor_args(tmp_path, 'cond_fam', '--sensor_device', 'device1')
    ds = StubSensorSet()
    ck = dict(meta=tr.checkpoint_meta(a, ds.sigma_d, 3e-7, 'psha', extra=ds.checkpoint_extra()))
    assert tr.check_resume_meta(ck, a, 0.02, 'psha', warn=False, sensor_sha='cfgsha')
    ck['meta']['sensor_device'] = 'device2'
    with pytest.raises(ValueError, match='sensor_device'):
        tr.check_resume_meta(ck, a, 0.02, 'psha', warn=False, sensor_sha='cfgsha')


def test_dry_run_handles_dict_batches(tmp_path, monkeypatch):
    a = sensor_args(tmp_path, 'cond_fam', '--dry_run_steps', '2')
    monkeypatch.setattr(tr, 'make_datasets', lambda args: (StubSensorSet(n=32), StubSensorSet()))
    rec = tr.dry_run(a)
    assert rec['steps'] == 2 and math.isfinite(rec['loss'])
    b = args_for(tmp_path, 'residual', '--dry_run_steps', '2', '--batch_size', '2')
    monkeypatch.setattr(tr, 'make_datasets', lambda args: (StubTupleSet(n=32), StubTupleSet()))
    assert math.isfinite(tr.dry_run(b)['loss'])
