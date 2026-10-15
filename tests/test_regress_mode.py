"""Mode 'regress' (Phase 1a-prime): the one-step residual regressor shares B's architecture, its target and its
training loop; only the objective differs (one deterministic call at t = REGRESS_T on a zero x_t placeholder)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import main_2d_residual_fsdp as tr
from model.regression_trainer import RegressionTrainer
from model.residual_wrapper import MODES, REGRESS_T, ResidualWrapper, build_residual_model, regress_forward, target_for

CPU = torch.device('cpu')


class LearnableResidualSet(torch.utils.data.Dataset):
    """(x64, y_d, x0_hat) HWC float32 on [-1, 1] whose residual r = (x - x0_hat) / sigma_d = 0.5 * x0_hat is a known
    target (training-loop smoke data)."""
    sigma_d = 0.05
    prior = {'s': 3e-7}

    def __init__(self, n=8, H=16, d=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.xh = torch.rand(n, H, H, 64, generator=g) * 1.6 - 0.8
        self.x = (self.xh + self.sigma_d * 0.5 * self.xh).clamp(-1, 1)
        self.y = torch.rand(n, H // d, H // d, 30, generator=g)
        self.img_list = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}_1.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        return self.x[i].numpy(), self.y[i].numpy(), self.xh[i].numpy()


def regress_args(tmp_path, **kw):
    a = tr.get_parser().parse_args(['--mode', 'regress', '--prediction_type', 'direct', '--loss_type', 'l2',
                                    '--device', 'cpu', '--base_channels', '8', '--batch_size', '2', '--num_workers', '0',
                                    '--num_epochs', '3', '--save_dir', str(tmp_path / 'run'), '--no_bf16'])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def parts_for(args, ds=None):
    ds = ds or LearnableResidualSet()
    loader = tr.make_loader(ds, args, batch_size=args.batch_size, shuffle=True, rank=0, world=1, device=CPU)
    model = build_residual_model('regress', args.base_channels)
    trainer = tr.make_trainer(args, CPU)
    opt = tr.make_optimizer(model, args)
    sched = tr.make_scheduler(opt, args)
    return ds, loader, model, trainer, opt, sched


def epoch(parts, args, e):
    ds, loader, model, trainer, opt, sched = parts
    tr.epoch_seed(args, e, 0)
    loader.sampler.set_epoch(e)
    rec = tr.train_one_epoch(model, trainer, loader, opt, args, CPU, tr.NO_DIST, 'regress', ds.sigma_d)
    sched.step()
    return rec


def test_regress_mode_shares_bs_architecture_and_target():
    assert MODES == ('residual', 'standard', 'standard_xhat', 'regress') and REGRESS_T == 999
    m = build_residual_model('regress', base_channels=8)
    assert isinstance(m, ResidualWrapper) and m.use_x0_hat and m.net.input_proj.in_channels == 128
    m.load_state_dict(build_residual_model('residual', base_channels=8).state_dict(), strict=True)   # only meta tells them apart
    x, xh = torch.rand(2, 4, 8, 8) * 2 - 1, torch.rand(2, 4, 8, 8) * 2 - 1
    assert torch.allclose(target_for('regress', x, xh, 0.05), (x - xh) / 0.05)


def test_regress_forward_uses_a_zero_placeholder_and_the_constant_t():
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen.update(x_t=x_t.clone(), t=t.clone(), keys=sorted(cond))
            return torch.ones_like(x_t)
    xh = torch.rand(3, 64, 16, 16)
    out = regress_forward(Spy(), {'sensor': torch.rand(3, 30, 4, 4), 'x0_hat': xh})
    assert out.shape == xh.shape
    assert torch.equal(seen['x_t'], torch.zeros_like(xh)) and seen['t'].dtype == torch.long
    assert seen['t'].tolist() == [REGRESS_T] * 3 and seen['keys'] == ['sensor', 'x0_hat']


@pytest.mark.parametrize('loss_type', ['l1', 'l2'])
def test_regression_trainer_loss_is_the_plain_per_sample_mean(loss_type):
    pred = torch.randn(2, 64, 16, 16)

    class Const(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return pred.clone()
    r = torch.randn(2, 64, 16, 16)
    cond = {'sensor': torch.rand(2, 30, 4, 4), 'x0_hat': torch.rand(2, 64, 16, 16)}
    trainer = RegressionTrainer(device=CPU, loss_type=loss_type)
    err = (pred - r).abs() if loss_type == 'l1' else (pred - r) ** 2
    torch.manual_seed(1)
    a, info = trainer.get_loss(Const(), r, cond, loss_in_fp32=True)
    torch.manual_seed(2)
    b, _ = trainer.get_loss(Const(), r, cond, loss_in_fp32=True)
    assert torch.allclose(a, err.mean()) and float(a) == float(b)                  # deterministic: no noise, no t draw
    assert trainer.prediction_type == 'direct' and trainer.last_nfe == 1 and info['t'] == REGRESS_T
    with pytest.raises(ValueError):
        RegressionTrainer(device=CPU, loss_type='huber')


def test_make_trainer_ties_mode_regress_to_prediction_type_direct(tmp_path):
    t = tr.make_trainer(regress_args(tmp_path), CPU)
    assert isinstance(t, RegressionTrainer) and t.loss_type == 'l2'
    for lt in ('l1', 'l2'):                 # --loss_type reaches the objective (l2 alone would match the default)
        assert tr.make_trainer(regress_args(tmp_path, loss_type=lt), CPU).loss_type == lt
    for bad in (dict(prediction_type='v'), dict(mode='residual')):                 # regress + v, residual + direct
        with pytest.raises(ValueError, match='direct'):
            tr.make_trainer(regress_args(tmp_path, **bad), CPU)
    with pytest.raises(ValueError, match='direct'):
        tr.check_mode_args(regress_args(tmp_path, prediction_type='x0'))


def test_regress_training_runs_and_its_loss_falls(tmp_path):
    """Smoke only: at base 8 the 8-channel input_proj cannot carry the 64 independent channels of r = 0.5 x0_hat, so
    the loss falls toward the x0_hat-blind floor 0.25 E[x0_hat^2], not below it. The training call itself is pinned
    by the spy test below; use of x0_hat by the y-swap Oracle test (Task 2)."""
    args = regress_args(tmp_path, lr=3e-3, num_epochs=8)
    torch.manual_seed(0)
    parts = parts_for(args)
    losses = [epoch(parts, args, e)['loss'] for e in range(8)]
    assert all(np.isfinite(losses)) and min(losses[-3:]) < 0.8 * losses[0], losses


def test_regress_training_step_calls_the_net_on_zero_x_t_at_regress_t_against_the_residual(tmp_path):
    seen = []

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x_t, cond, t):
            seen.append((x_t.detach().clone(), t.clone()))
            return torch.zeros_like(x_t) + self.p
    args = regress_args(tmp_path, lr=0.0)
    ds = LearnableResidualSet()
    loader = tr.make_loader(ds, args, batch_size=2, shuffle=True, rank=0, world=1, device=CPU)
    model = Spy()
    tr.epoch_seed(args, 0, 0)
    loader.sampler.set_epoch(0)
    rec = tr.train_one_epoch(model, tr.make_trainer(args, CPU), loader, tr.make_optimizer(model, args), args, CPU,
                             tr.NO_DIST, 'regress', ds.sigma_d)
    assert len(seen) == rec['steps'] == 4
    assert all(float(x.abs().max()) == 0.0 and t.tolist() == [REGRESS_T] * 2 for x, t in seen)
    assert rec['loss'] == pytest.approx(0.25 * float((ds.xh ** 2).mean()), rel=1e-4)   # zero predictor vs r = 0.5 x0_hat


def test_regress_validate_is_deterministic(tmp_path):
    args = regress_args(tmp_path)
    ds, _, model, trainer, _, _ = parts_for(args)
    vl = tr.make_loader(ds, args, batch_size=2, shuffle=False, rank=0, world=1, device=CPU)
    a = tr.validate(model, trainer, vl, args, CPU, tr.NO_DIST, 'regress', ds.sigma_d)
    torch.manual_seed(5)
    torch.rand(4)
    b = tr.validate(model, trainer, vl, args, CPU, tr.NO_DIST, 'regress', ds.sigma_d)
    assert a['val_loss'] == b['val_loss'] and a['val_n'] == 8


def test_regress_checkpoint_meta_is_refused_as_residual(tmp_path):
    from scripts.eval_warmstart import check_meta
    args = regress_args(tmp_path)
    parts = parts_for(args)
    ds, _, model, _, opt, sched = parts
    epoch(parts, args, 0)
    sha = 'ab' * 32
    meta = tr.checkpoint_meta(args, ds.sigma_d, ds.prior['s'], sha)
    assert meta['mode'] == 'regress' and meta['prediction_type'] == 'direct' and meta['args']['loss_type'] == 'l2'
    path = str(tmp_path / 'run' / 'checkpoint_epoch_1.pth')
    tr.save_checkpoint(model, opt, sched, 0, 0.5, meta, path, 0.01, tr.NO_DIST)
    ck = tr.load_checkpoint(path)
    assert tr.check_resume_meta(ck, args, ds.sigma_d, sha, warn=False) is True
    with pytest.raises(ValueError, match='mode'):
        tr.check_resume_meta(ck, regress_args(tmp_path, mode='residual', prediction_type='v'), ds.sigma_d, sha, warn=False)
    with pytest.raises(ValueError, match='loss_type'):          # the L1 and L2 runs differ only in --loss_type
        tr.check_resume_meta(ck, regress_args(tmp_path, loss_type='l1'), ds.sigma_d, sha, warn=False)
    with pytest.raises(ValueError, match='seed'):
        tr.check_resume_meta(ck, regress_args(tmp_path, seed=43), ds.sigma_d, sha, warn=False)
    eval_args = SimpleNamespace(ckpt=path, mode='residual', d=4, base_channels=8, prediction_type='v')
    with pytest.raises(ValueError, match="mode: checkpoint 'regress'"):
        check_meta(ck, eval_args, ds.sigma_d, sha)
    assert check_meta(ck, SimpleNamespace(**dict(vars(eval_args), mode='regress', prediction_type='direct')),
                      ds.sigma_d, sha) is True
