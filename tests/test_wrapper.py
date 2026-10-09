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
