import torch

from model.u2net_hyperspectral import U2NetHyperspectral


def test_default_signature_unchanged():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    assert net.input_proj.in_channels == 8
    out = net(torch.randn(2, 8, 16, 16), torch.randn(2, 5, 16, 16), torch.randint(0, 1000, (2,)))
    assert out.shape == (2, 8, 16, 16)


def test_extra_in_channels_widens_input_only():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16, extra_in_channels=8)
    assert net.input_proj.in_channels == 16
    out = net(torch.randn(2, 16, 16, 16), torch.randn(2, 5, 16, 16), torch.randint(0, 1000, (2,)))
    assert out.shape == (2, 8, 16, 16)


def test_old_state_dict_still_loads_strict():
    a = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    b = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    b.load_state_dict(a.state_dict(), strict=True)
