"""CPU tests of the training entry point on a tiny synthetic dataset (no HFD files, no GPU)."""
import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

import main_2d_residual_fsdp as tr
from model.residual_wrapper import build_residual_model

CPU = torch.device('cpu')


class TinyResidualSet(torch.utils.data.Dataset):
    """Stands in for HFDResidualData: (x64, y_d, x0_hat) as HWC float32 numpy arrays on the [-1, 1] scale."""
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


def tiny_args(tmp_path, **kw):
    a = tr.get_parser().parse_args(['--mode', 'residual', '--device', 'cpu', '--base_channels', '8', '--batch_size', '2',
                                    '--num_workers', '0', '--num_epochs', '3', '--save_dir', str(tmp_path / 'run'), '--no_bf16'])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def make_parts(args, mode='residual', ds=None):
    ds = ds or TinyResidualSet()
    loader = tr.make_loader(ds, args, batch_size=args.batch_size, shuffle=True, rank=0, world=1, device=CPU)
    model = build_residual_model(mode, args.base_channels)
    trainer = tr.make_trainer(args, CPU)
    opt = tr.make_optimizer(model, args)
    sched = tr.make_scheduler(opt, args)
    return ds, loader, model, trainer, opt, sched


def one_epoch(parts, args, epoch=0, mode='residual'):
    ds, loader, model, trainer, opt, sched = parts
    tr.epoch_seed(args, epoch, 0); loader.sampler.set_epoch(epoch)
    rec = tr.train_one_epoch(model, trainer, loader, opt, args, CPU, tr.NO_DIST, mode, ds.sigma_d)
    sched.step()
    return rec


def test_train_one_epoch_runs_and_reports(tmp_path):
    args = tiny_args(tmp_path)
    rec = one_epoch(make_parts(args), args)
    assert set(rec) >= {'loss', 'lr', 'steps', 'skipped_nonfinite', 'sec', 'data_sec'}
    assert rec['steps'] == 4 and rec['skipped_nonfinite'] == 0 and torch.isfinite(torch.tensor(rec['loss']))


def test_cond_dict_keys_reach_get_loss(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    parts = make_parts(args); trainer = parts[3]
    seen = {}
    real = trainer.get_loss

    def spy(m, x0, cond, **kw):
        seen['keys'] = sorted(cond); seen['kw'] = kw
        return real(m, x0, cond, **kw)
    monkeypatch.setattr(trainer, 'get_loss', spy)
    one_epoch(parts, args)
    assert seen['keys'] == ['sensor', 'x0_hat'] and seen['kw'] == {'loss_in_fp32': True}


@pytest.mark.parametrize('mode', ['residual', 'standard', 'standard_xhat'])
def test_target_per_mode_reaches_get_loss(tmp_path, monkeypatch, mode):
    args = tiny_args(tmp_path, mode=mode)
    parts = make_parts(args, mode); trainer = parts[3]
    got = {}
    real = trainer.get_loss

    def spy(m, x0, cond, **kw):
        got.setdefault('x0', x0.clone())
        return real(m, x0, cond, **kw)
    monkeypatch.setattr(trainer, 'get_loss', spy)
    one_epoch(parts, args, mode=mode)
    assert (got['x0'].abs().max() > 1.5) if mode == 'residual' else (got['x0'].abs().max() <= 1.0)


def test_save_load_roundtrip_meta_and_harness_check(tmp_path):
    from scripts.eval_warmstart import check_meta
    args = tiny_args(tmp_path)
    parts = make_parts(args); ds, _, model, _, opt, sched = parts
    one_epoch(parts, args)
    meta = tr.checkpoint_meta(args, ds.sigma_d, ds.prior['s'], 'deadbeef' * 8)
    path = tmp_path / 'run' / 'checkpoint_epoch_1.pth'
    tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, str(path), 0.01, tr.NO_DIST)
    assert path.exists() and not [f for f in os.listdir(path.parent) if f.endswith('.tmp')]
    ck = tr.load_checkpoint(str(path))
    assert set(ck) >= {'epoch', 'loss', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'meta', 'elapsed_hours'}
    assert all(k.startswith('net.') for k in ck['model_state_dict'])
    assert ck['meta'] == dict(mode='residual', d=4, base_channels=8, prediction_type='v', n_timesteps=1000,
                              sigma_d=ds.sigma_d, prior_s=3e-7, prior_sha256='deadbeef' * 8, args=vars(args))
    fresh = build_residual_model('residual', 8)
    fresh.load_state_dict(ck['model_state_dict'], strict=True)
    assert all(torch.equal(a, b) for a, b in zip(fresh.parameters(), model.parameters()))
    eval_args = SimpleNamespace(ckpt=str(path), mode='residual', d=4, base_channels=8, prediction_type='v')
    assert check_meta(ck, eval_args, ds.sigma_d, 'deadbeef' * 8) is True            # the harness accepts what training wrote
    with pytest.raises(ValueError, match='base_channels'):
        check_meta(ck, SimpleNamespace(**dict(vars(eval_args), base_channels=16)), ds.sigma_d, 'deadbeef' * 8)


def test_save_checkpoint_is_atomic(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    ds, _, model, _, opt, sched = make_parts(args)
    meta = tr.checkpoint_meta(args, ds.sigma_d, 3e-7, 'x' * 64)
    path = str(tmp_path / 'run' / 'checkpoint_epoch_1.pth')
    tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, path, 0.0, tr.NO_DIST)
    good = open(path, 'rb').read()

    def broken_save(obj, f, **kw):
        open(f, 'wb').write(b'partial'); raise OSError('disk full')
    monkeypatch.setattr(tr.torch, 'save', broken_save)
    with pytest.raises(OSError):
        tr.save_checkpoint(model, opt, sched, 1, 0.4, meta, path, 0.1, tr.NO_DIST)
    assert open(path, 'rb').read() == good                                          # the old checkpoint survives


def test_resume_is_bitwise_exact(tmp_path):
    """2 epochs + save + fresh process + restore + 1 epoch == 3 uninterrupted epochs (Review Focus 3)."""
    args = tiny_args(tmp_path, num_epochs=3)
    torch.manual_seed(0)
    parts = make_parts(args); ds, _, model, _, opt, sched = parts
    for e in range(2):
        one_epoch(parts, args, e)
    path = str(tmp_path / 'run' / 'checkpoint_epoch_2.pth')
    tr.save_checkpoint(model, opt, sched, 1, 0.4, tr.checkpoint_meta(args, ds.sigma_d, 3e-7, 'x' * 64), path, 0.2, tr.NO_DIST)
    one_epoch(parts, args, 2)                                                        # the uninterrupted third epoch
    ref = [p.detach().clone() for p in model.parameters()]
    ck = tr.load_checkpoint(path)
    parts2 = make_parts(args); ds2, _, model2, _, opt2, sched2 = parts2
    model2.load_state_dict(ck['model_state_dict'], strict=True)
    start_epoch, elapsed = tr.restore_training_state(ck, model2, opt2, sched2, tr.NO_DIST)
    assert start_epoch == 2 and elapsed == 0.2 and sched2.last_epoch == 2
    one_epoch(parts2, args, start_epoch)
    assert all(torch.equal(a, b) for a, b in zip(ref, model2.parameters()))


def test_prune_keeps_last_two_and_milestones(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    for e in (10, 20, 25, 30, 40, 50, 60):
        (run / f'checkpoint_epoch_{e}.pth').write_bytes(b'x')
    (run / 'args.json').write_text('{}')
    tr.prune_checkpoints(str(run), keep_last=2, keep_epochs=(25, 50, 100))
    assert sorted(f for f in os.listdir(run) if f.endswith('.pth')) == ['checkpoint_epoch_25.pth', 'checkpoint_epoch_50.pth', 'checkpoint_epoch_60.pth']


def test_nonfinite_loss_is_skipped_without_a_step_then_aborts(tmp_path, monkeypatch):
    args = tiny_args(tmp_path, max_nonfinite=10)
    parts = make_parts(args); ds, loader, model, trainer, opt, sched = parts
    before = [p.detach().clone() for p in model.parameters()]
    real = trainer.get_loss; calls = {'n': 0}

    def nan_first(m, x0, cond, **kw):
        calls['n'] += 1
        return (torch.tensor(float('nan'), requires_grad=True), {}) if calls['n'] == 1 else real(m, x0, cond, **kw)
    monkeypatch.setattr(trainer, 'get_loss', nan_first)
    rec = one_epoch(parts, args)
    assert rec['skipped_nonfinite'] == 1 and rec['steps'] == 3
    monkeypatch.setattr(trainer, 'get_loss', lambda m, x0, cond, **kw: (torch.tensor(float('nan'), requires_grad=True), {}))
    args.max_nonfinite = 2
    with pytest.raises(RuntimeError, match='non-finite'):
        one_epoch(parts, args, 1)
    # an all-NaN epoch shorter than max_nonfinite must not touch the parameters
    parts3 = make_parts(args); model3 = parts3[2]; before3 = [p.detach().clone() for p in model3.parameters()]
    args.max_nonfinite = 100
    monkeypatch.setattr(parts3[3], 'get_loss', lambda m, x0, cond, **kw: (torch.tensor(float('nan'), requires_grad=True), {}))
    assert one_epoch(parts3, args)['steps'] == 0
    assert all(torch.equal(a, b) for a, b in zip(before3, model3.parameters()))


def test_nonfinite_skip_is_collective(tmp_path, monkeypatch):
    """A finite loss on this rank is still skipped when the all-reduced flag says another rank was non-finite."""
    args = tiny_args(tmp_path)
    parts = make_parts(args); ds, loader, model, trainer, opt, sched = parts
    before = [p.detach().clone() for p in model.parameters()]
    fake_ctx = SimpleNamespace(enabled=True, rank=0, world=2, local_rank=0)

    def all_reduce(t, op=None):
        t.zero_()                                            # "some rank saw NaN"
    monkeypatch.setattr(tr.dist, 'all_reduce', all_reduce)
    monkeypatch.setattr(tr.dist, 'ReduceOp', SimpleNamespace(MIN=0, SUM=1))
    tr.epoch_seed(args, 0, 0); loader.sampler.set_epoch(0)
    rec = tr.train_one_epoch(model, trainer, loader, opt, args, CPU, fake_ctx, 'residual', ds.sigma_d)
    assert rec['steps'] == 0 and rec['skipped_nonfinite'] == 4
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_save_checkpoint_gathers_on_every_rank(tmp_path, monkeypatch):
    """Under FSDP the state-dict gathers and the barrier must run on every rank; only the write is rank-gated."""
    args = tiny_args(tmp_path)
    ds, _, model, _, opt, sched = make_parts(args)
    events = []

    class FakeFSDP:
        @staticmethod
        @contextlib.contextmanager
        def state_dict_type(m, t, *cfgs):
            events.append('gather'); yield

        @staticmethod
        def optim_state_dict(m, o):
            events.append('optim'); return o.state_dict()
    monkeypatch.setattr(tr, 'FSDP', FakeFSDP)
    monkeypatch.setattr(tr.dist, 'barrier', lambda: events.append('barrier'))
    meta = tr.checkpoint_meta(args, ds.sigma_d, 3e-7, 'x' * 64)
    for rank in (1, 0):
        ctx = SimpleNamespace(enabled=True, rank=rank, world=2, local_rank=rank)
        tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, str(tmp_path / f'r{rank}.pth'), 0.0, ctx)
    assert events == ['gather', 'optim', 'barrier', 'gather', 'optim', 'barrier']
    assert not (tmp_path / 'r1.pth').exists() and (tmp_path / 'r0.pth').exists()


def test_run_epochs_calls_save_on_every_rank(tmp_path, monkeypatch):
    """The call site in the epoch loop is not rank-gated (fake rank 1 still reaches save_checkpoint)."""
    args = tiny_args(tmp_path, num_epochs=1, save_every=1, keep_epochs=[], val_every=0)
    parts = make_parts(args)
    calls = []
    monkeypatch.setattr(tr, 'save_checkpoint', lambda *a, **k: calls.append(a[6]))
    fake_ctx = SimpleNamespace(enabled=False, rank=1, world=2, local_rank=1)         # enabled=False: no collectives on CPU
    tr.run_epochs(args, parts, fake_ctx, CPU)
    assert calls and calls[0].endswith('checkpoint_epoch_1.pth')
    assert not os.path.exists(tmp_path / 'run' / 'train_log.jsonl')                 # rank 1 writes no logs


def test_validate_is_deterministic_and_reduced(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    ds, _, model, trainer, _, _ = make_parts(args)
    vloader = tr.make_loader(ds, args, batch_size=2, shuffle=False, rank=0, world=1, device=CPU)
    a = tr.validate(model, trainer, vloader, args, CPU, tr.NO_DIST, 'residual', ds.sigma_d)
    torch.manual_seed(999); torch.rand(3)
    b = tr.validate(model, trainer, vloader, args, CPU, tr.NO_DIST, 'residual', ds.sigma_d)
    assert a['val_loss'] == pytest.approx(b['val_loss']) and a['val_n'] == 8
    monkeypatch.setattr(tr.dist, 'all_reduce', lambda t, op=None: t.mul_(2))           # "the other rank saw the same"
    monkeypatch.setattr(tr.dist, 'ReduceOp', SimpleNamespace(MIN=0, SUM=1))
    c = tr.validate(model, trainer, vloader, args, CPU, SimpleNamespace(enabled=True, rank=0, world=2, local_rank=0), 'residual', ds.sigma_d)
    assert c['val_n'] == 16 and c['val_loss'] == pytest.approx(a['val_loss'])


def test_milestones_schedule(tmp_path):
    args = tiny_args(tmp_path, num_epochs=100, save_every=10, keep_epochs=[25, 50, 100])
    assert [e for e in range(1, 101) if tr.should_save(e, args)] == sorted(set(list(range(10, 101, 10)) + [25, 50, 100]))


def test_make_datasets_excludes_gate_cubes_from_validation(tmp_path, monkeypatch):
    """The monitoring loss never sees the 200 gate cubes (no peeking)."""
    import json
    sub = json.load(open(os.path.join(tr.REPO, 'tests', 'data', 'val_subset_200.json')))
    fake_train = TinyResidualSet(n=4); fake_val = TinyResidualSet(n=8)
    assert len(sub['files']) == 200
    fake_val.img_list = [os.path.join('/root', f) for f in sub['files']] + ['/root/MatFlower60/Train/P023/other.mat'] * 3
    monkeypatch.setattr(tr, 'HFDResidualData', lambda root, split, **kw: fake_train if split == 'train' else fake_val)
    args = tiny_args(tmp_path, data_root='/root')
    train, val = tr.make_datasets(args)
    assert len(train) == 4 and len(val) == 3 and all('other' in f for f in val.img_list)


def test_make_datasets_fails_loudly_when_gate_files_are_missing_from_validation(tmp_path, monkeypatch):
    """A path-form mismatch (gate files not recognised in the validation list) must not silently peek."""
    import json
    sub = json.load(open(os.path.join(tr.REPO, 'tests', 'data', 'val_subset_200.json')))
    fake_val = TinyResidualSet(n=8)
    fake_val.img_list = [os.path.join('/root', f) for f in sub['files'][:5]] + ['/root/MatFlower60/Train/P023/other.mat'] * 3
    monkeypatch.setattr(tr, 'HFDResidualData', lambda root, split, **kw: TinyResidualSet(n=4) if split == 'train' else fake_val)
    with pytest.raises(AssertionError, match=r'only 5 of 200 gate files'):
        tr.make_datasets(tiny_args(tmp_path, data_root='/root'))


# ----------------------------------------------------------------------------- resume refuses a mismatched checkpoint
def saved_checkpoint(tmp_path, **arg_kw):
    """A checkpoint written by a (tiny) run with the default arguments; returns (ck, path, args, ds)."""
    args = tiny_args(tmp_path, **arg_kw)
    parts = make_parts(args); ds, _, model, _, opt, sched = parts
    path = str(tmp_path / 'run' / 'checkpoint_epoch_1.pth')
    tr.save_checkpoint(model, opt, sched, 0, 0.5, tr.checkpoint_meta(args, ds.sigma_d, 3e-7, 'x' * 64), path, 0.0, tr.NO_DIST)
    return tr.load_checkpoint(path), path, args, ds


def test_resume_meta_matching_passes(tmp_path):
    ck, path, args, ds = saved_checkpoint(tmp_path)
    assert tr.check_resume_meta(ck, args, ds.sigma_d, 'x' * 64, path=path) is True


def test_resume_meta_mode_mismatch_raises(tmp_path):
    """A' and B have identical parameter shapes, so only the meta check can catch a mode switch."""
    ck, path, args, ds = saved_checkpoint(tmp_path)
    build_residual_model('standard_xhat', 8).load_state_dict(ck['model_state_dict'], strict=True)      # strict load alone is blind to it
    args.mode = 'standard_xhat'
    with pytest.raises(ValueError, match='mode'):
        tr.check_resume_meta(ck, args, ds.sigma_d, 'x' * 64, path=path)


def test_resume_meta_names_every_mismatch(tmp_path):
    ck, path, args, ds = saved_checkpoint(tmp_path)
    args.d, args.base_channels, args.prediction_type = 8, 16, 'eps'
    with pytest.raises(ValueError) as e:
        tr.check_resume_meta(ck, args, ds.sigma_d + 1e-3, 'y' * 64, path=path)
    msg = str(e.value)
    assert all(k in msg for k in ('d: checkpoint 4 vs args 8', 'base_channels', 'prediction_type', 'sigma_d', 'prior_sha256')) and path in msg


def test_resume_meta_d_mismatch_raises(tmp_path):
    ck, path, args, ds = saved_checkpoint(tmp_path)
    args.d = 8
    with pytest.raises(ValueError, match='d: checkpoint 4 vs args 8'):
        tr.check_resume_meta(ck, args, ds.sigma_d, 'x' * 64, path=path)


def test_resume_meta_prior_hash_mismatch_raises(tmp_path):
    ck, path, args, ds = saved_checkpoint(tmp_path)
    with pytest.raises(ValueError, match='prior_sha256'):
        tr.check_resume_meta(ck, args, ds.sigma_d, 'a different prior', path=path)


def test_resume_meta_missing_meta_is_refused(tmp_path):
    ck, path, args, ds = saved_checkpoint(tmp_path)
    del ck['meta']
    with pytest.raises(ValueError, match='no meta'):
        tr.check_resume_meta(ck, args, ds.sigma_d, 'x' * 64, path=path)


def test_resume_t_max_mismatch_warns_but_does_not_raise(tmp_path, capsys):
    ck, path, args, ds = saved_checkpoint(tmp_path, num_epochs=3)
    args.num_epochs = 5
    capsys.readouterr()
    assert tr.check_resume_meta(ck, args, ds.sigma_d, 'x' * 64, path=path) is True
    out = capsys.readouterr().out
    assert 'WARNING' in out and 'T_max=3' in out and '--num_epochs 5' in out
    args.num_epochs = 3
    assert tr.check_resume_meta(ck, args, ds.sigma_d, 'x' * 64, path=path) is True
    assert capsys.readouterr().out == ''                                              # no warning when they agree


def test_build_everything_checks_resume_meta(tmp_path, monkeypatch):
    """The wiring: build_everything itself refuses a mismatched --resume and accepts a matching one."""
    ck, path, args, ds = saved_checkpoint(tmp_path)
    monkeypatch.setattr(tr, 'make_datasets', lambda a: (TinyResidualSet(), TinyResidualSet()))
    monkeypatch.setattr(tr, 'file_sha256', lambda p: 'x' * 64)
    args.resume = path
    parts = tr.build_everything(args, tr.NO_DIST, CPU)
    assert parts.start_epoch == 1 and parts.resume_path == path
    args.mode = 'standard_xhat'
    with pytest.raises(ValueError, match='mode'):
        tr.build_everything(args, tr.NO_DIST, CPU)
