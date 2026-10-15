import torch

from model.residual_wrapper import ResidualWrapper
from model.u2net_hyperspectral import U2NetHyperspectral


def _cond(B=2, H=16, d=4):
    return {'sensor': torch.randn(B, 5, H // d, H // d), 'x0_hat': torch.randn(B, 8, H, H)}


def test_residual_mode_concatenates_x0_hat():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16, extra_in_channels=8)
    m = ResidualWrapper(net, use_x0_hat=True)
    out = m(torch.randn(2, 8, 16, 16), _cond(), torch.randint(0, 1000, (2,)))
    assert out.shape == (2, 8, 16, 16)


def test_standard_mode_ignores_x0_hat():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    m = ResidualWrapper(net, use_x0_hat=False)
    x_t, t = torch.randn(2, 8, 16, 16), torch.randint(0, 1000, (2,))
    c1, c2 = _cond(), _cond()
    c2['sensor'] = c1['sensor']                      # same sensor, different x0_hat
    torch.manual_seed(0); o1 = m(x_t, c1, t)
    torch.manual_seed(0); o2 = m(x_t, c2, t)
    assert torch.allclose(o1, o2)


def test_residual_mode_uses_x0_hat():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16, extra_in_channels=8)
    m = ResidualWrapper(net, use_x0_hat=True).eval()
    x_t, t = torch.randn(2, 8, 16, 16), torch.randint(0, 1000, (2,))
    c1, c2 = _cond(), _cond()
    c2['sensor'] = c1['sensor']
    with torch.no_grad():
        assert not torch.allclose(m(x_t, c1, t), m(x_t, c2, t))


def test_wrapper_rejects_wrong_channel_count():
    import pytest
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)   # no extra channels
    m = ResidualWrapper(net, use_x0_hat=True)
    with pytest.raises(ValueError):
        m(torch.randn(2, 8, 16, 16), _cond(), torch.randint(0, 1000, (2,)))


import pytest
import torch

from model.residual_wrapper import MODES, ResidualWrapper, build_residual_model, target_for


@pytest.mark.parametrize('mode,in_ch,use_xhat', [('residual', 128, True), ('standard', 64, False), ('standard_xhat', 128, True)])
def test_build_residual_model_per_mode(mode, in_ch, use_xhat):
    m = build_residual_model(mode, base_channels=8)
    assert isinstance(m, ResidualWrapper) and m.use_x0_hat is use_xhat
    assert m.net.input_proj.in_channels == in_ch
    assert all(k.startswith('net.') for k in m.state_dict())
    x_t = torch.randn(1, 64, 16, 16)
    cond = {'sensor': torch.randn(1, 30, 4, 4), 'x0_hat': torch.randn(1, 64, 16, 16)}
    assert m(x_t, cond, torch.tensor([3])).shape == x_t.shape


def test_build_residual_model_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_residual_model('standard_x0hat', base_channels=8)
    assert MODES == ('residual', 'standard', 'standard_xhat', 'regress')


def test_target_for_modes():
    x = torch.rand(2, 4, 8, 8) * 2 - 1
    xh = torch.rand(2, 4, 8, 8) * 2 - 1
    assert torch.allclose(target_for('residual', x, xh, 0.05), (x - xh) / 0.05)
    assert torch.equal(target_for('standard', x, xh, 0.05), x)
    assert torch.equal(target_for('standard_xhat', x, xh, 0.05), x)
