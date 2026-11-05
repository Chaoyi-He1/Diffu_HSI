# tests/test_sensor_modes.py
import torch
from model import residual_wrapper as RW
from model.u2net_hyperspectral import U2NetHyperspectral

def test_mode_tables():
    assert set(RW.FAM_MODES) <= set(RW.MODES) and 'uncond' in RW.MODES and set(RW.FAM_MODES) <= set(RW.CENTRED_MODES)

def test_u2net_runs_without_context_and_depends_on_t():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    with torch.no_grad():                       # FiLM layers are zero-initialised: without a context t reaches the net only through
        for n, p in net.named_parameters():     # them, so a fresh net is t-blind by construction; perturb them before the check
            if 'film' in n.lower():
                p.normal_(0.0, 0.1)
    x = torch.randn(2, 8, 16, 16)
    a = net(x, None, torch.tensor([10, 10])); b = net(x, None, torch.tensor([900, 900]))
    assert a.shape == (2, 8, 16, 16) and not torch.allclose(a, b)

def test_uncond_and_fam_models_shapes():
    u = RW.build_residual_model('uncond', 16, spectral_channels=8, sensor_channels=5)
    assert u.net.input_proj.in_channels == 8 and u(torch.randn(2, 8, 16, 16), {'sensor': None}, torch.zeros(2, dtype=torch.long)).shape == (2, 8, 16, 16)
    c = RW.build_residual_model('cond_fam', 16, spectral_channels=8, sensor_channels=5)
    cond = {'sensor': None, 'x0_hat': torch.randn(2, 8, 16, 16), 's_post': torch.rand(2, 8, 16, 16) + 0.02}
    assert c.net.input_proj.in_channels == 24 and c(torch.randn(2, 8, 16, 16), cond, torch.zeros(2, dtype=torch.long)).shape == (2, 8, 16, 16)
    r = RW.build_residual_model('regress_fam', 16, spectral_channels=8, sensor_channels=5)
    assert RW.regress_forward(r, cond).shape == (2, 8, 16, 16) and RW.varhead_forward(RW.build_residual_model('varhead_fam', 16, 8, 5), cond).shape == (2, 8, 16, 16)

def test_legacy_modes_unchanged():
    b = RW.build_residual_model('residual', 16, spectral_channels=8, sensor_channels=5)
    assert b.net.input_proj.in_channels == 16 and b.use_context and not b.use_s_post
    b2 = RW.build_residual_model('residual', 16, spectral_channels=8, sensor_channels=5); b2.load_state_dict(b.state_dict(), strict=True)

def test_target_for_accepts_tensor_scale_and_uncond_returns_x():
    x, xh, s = torch.ones(1, 2, 2, 2), torch.zeros(1, 2, 2, 2), torch.full((1, 2, 2, 2), 0.5)
    assert torch.equal(RW.target_for('cond_fam', x, xh, s), 2 * torch.ones_like(x)) and torch.equal(RW.target_for('uncond', x, None, None), x)

def test_batch_to_cond_both_batch_kinds():
    x, cond, scale = RW.batch_to_cond('residual', (torch.zeros(1, 4, 2, 2), torch.zeros(1, 3, 1, 1), torch.zeros(1, 4, 2, 2)), 0.05, torch.device('cpu'))
    assert set(cond) == {'sensor', 'x0_hat'} and scale == 0.05
    b = dict(x=torch.zeros(1, 4, 2, 2), y=torch.zeros(1, 3, 2, 2), x0_hat=torch.zeros(1, 4, 2, 2), s_post=torch.full((1, 4, 2, 2), 0.1), s_raw=torch.zeros(1, 4, 2, 2), mask=torch.ones(1, 3, 2, 2), sigma=torch.zeros(1), snr_db=torch.zeros(1), snapshot=torch.zeros(1))
    x, cond, scale = RW.batch_to_cond('cond_fam', b, 0.02, torch.device('cpu')); assert cond['sensor'] is None and torch.equal(scale, b['s_post'])
    x, cond, scale = RW.batch_to_cond('uncond', b, 0.02, torch.device('cpu')); assert cond == {'sensor': None} and scale is None

def test_uncond_tt_uses_time_tokens():
    # fresh (zero-initialised) FiLM: t can reach the net only through the cross-attention time tokens
    x = torch.randn(2, 8, 16, 16)
    torch.manual_seed(0); tt = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16, time_tokens_when_no_context=True)
    torch.manual_seed(0); plain = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    assert tt.time_tokens_when_no_context and not plain.time_tokens_when_no_context
    assert tt.state_dict().keys() == plain.state_dict().keys()
    assert all(torch.equal(v, plain.state_dict()[k]) for k, v in tt.state_dict().items())     # the flag adds no parameters
    t10, t900 = torch.tensor([10, 10]), torch.tensor([900, 900])
    with torch.no_grad():
        assert not torch.allclose(tt(x, None, t10), tt(x, None, t900))
        assert torch.equal(plain(x, None, t10), plain(x, None, t900))
    m = RW.build_residual_model('uncond_tt', 16, 8, 5)
    u = RW.build_residual_model('uncond', 16, 8, 5)
    assert m.net.input_proj.in_channels == 8 and not m.use_x0_hat and not m.use_s_post and not m.use_context
    assert m.net.time_tokens_when_no_context and not u.net.time_tokens_when_no_context
    assert m.state_dict().keys() == u.state_dict().keys()
    m.load_state_dict(u.state_dict(), strict=True)
    assert m(x, {'sensor': None}, t10).shape == (2, 8, 16, 16)
    assert 'uncond_tt' in RW.MODES and RW.MODES[-1] == 'uncond_tt' and 'uncond_tt' in RW.SENSOR_MODES
    assert 'uncond_tt' not in RW.CENTRED_MODES and 'uncond_tt' not in RW.ONE_CALL_MODES
    assert torch.equal(RW.target_for('uncond_tt', x, None, None), x)
    b = dict(x=torch.zeros(1, 4, 2, 2), y=torch.zeros(1, 3, 2, 2), x0_hat=torch.zeros(1, 4, 2, 2), s_post=torch.full((1, 4, 2, 2), 0.1))
    xb, cond, scale = RW.batch_to_cond('uncond_tt', b, 0.02, torch.device('cpu'))
    assert cond == {'sensor': None} and scale is None and torch.equal(xb, b['x'])
