import numpy as np, pytest, torch
from data_loader import hfd_sensor_sim as DS
from data_loader.sensor_config import default_training_fields
from data_loader.sensor_forward import mosaic_mask

def _cfg(): return dict(default_training_fields('p' * 64), P_y=50.0)

@pytest.fixture
def ds(fake_sensor_root, synthetic_prior, tmp_path):
    np.savez(tmp_path / 'prior.npz', mu=synthetic_prior['mu'], C=synthetic_prior['C'], s=3e-7)
    return DS.HFDSensorSim(fake_sensor_root, 'train', _cfg(), device='device1', prior_path=str(tmp_path / 'prior.npz'))

def test_item_keys_shapes_and_scales(ds):
    it = ds[0]
    assert set(it) == {'x', 'y', 'x0_hat', 's_post', 's_raw', 'mask', 'sigma', 'snr_db', 'snapshot'}
    assert it['x'].shape == it['x0_hat'].shape == it['s_post'].shape == (16, 16, 64) and it['y'].shape == it['mask'].shape == (16, 16, 30)
    assert it['x'].dtype == np.float32 and -1 <= it['x'].min() and it['x'].max() <= 1
    assert it['s_post'].min() >= 2 * 0.01 - 1e-6 and (it['s_raw'] <= it['s_post'] + 1e-6).all()

def test_y_is_R_transpose_x01_plus_seeded_noise(ds):
    ds_fixed = DS.HFDSensorSim(ds.data_root, 'train', _cfg(), setting=dict(mode='sequential', snr_db=60.0), prior_path=ds.prior_path)
    it = ds_fixed[1]; x01 = DS.load_cube01(ds_fixed.img_list[1]); clean = x01 @ ds_fixed.R
    assert np.abs(it['y'] - clean).max() < 6 * it['sigma'] + 1e-6 and not np.allclose(it['y'], clean)
    np.testing.assert_array_equal(ds_fixed[1]['y'], it['y'])                      # same epoch, same idx -> same noise

def test_set_epoch_changes_training_noise_but_not_the_cube(ds):
    a = ds[2]; ds.set_epoch(1); b = ds[2]
    np.testing.assert_array_equal(a['x'], b['x']); assert not np.array_equal(a['y'], b['y'])

def test_family_frequencies(ds):
    draws = []
    for e in range(167):                                                         # 167 epochs x 12 files: the rngs are keyed by file
        ds.set_epoch(e); draws += [ds.draw_setting(i) for i in range(len(ds))]
    assert abs(np.mean([m == 'snapshot' for m, _ in draws]) - 0.5) < 0.04
    assert abs(np.mean([s == 60.0 for _, s in draws]) - 0.1) < 0.03
    others = [s for _, s in draws if s != 60.0]; assert min(others) >= 20.0 and max(others) <= 50.0

def test_noise_depends_on_the_file_not_its_position(ds):
    rev = DS.HFDSensorSim(ds.data_root, 'train', _cfg(), device='device1', prior_path=ds.prior_path)
    rev.img_list = list(reversed(ds.img_list))
    n = len(ds)
    for i in (0, 1, n - 1):
        j = n - 1 - i
        assert rev.img_list[j] == ds.img_list[i] and rev.file_key(j) == ds.file_key(i)
        assert rev.draw_setting(j) == ds.draw_setting(i)
        a, b = ds[i], rev[j]
        assert a['y'].tobytes() == b['y'].tobytes() and a['sigma'] == b['sigma'] and a['snapshot'] == b['snapshot']
    assert ds.checkpoint_extra()['seed_scheme'] == 'crc32(relpath)'

def test_snapshot_item_mask_and_zero_off_mask(fake_sensor_root, ds):
    d2 = DS.HFDSensorSim(fake_sensor_root, 'train', _cfg(), setting=dict(mode='snapshot', snr_db=30.0), prior_path=ds.prior_path)
    it = d2[0]; np.testing.assert_array_equal(it['mask'], mosaic_mask(16, 16)); assert np.all(it['y'][it['mask'] == 0] == 0) and it['snapshot'] == 1

def test_collate_and_extra(ds):
    b = DS.sensor_collate_fn([ds[0], ds[1]])
    assert b['x'].shape == (2, 64, 16, 16) and b['y'].shape == (2, 30, 16, 16) and b['sigma'].shape == (2,)
    e = ds.checkpoint_extra(); assert e['sensor_config_sha'] == ds.config_sha and e['sensor_device'] == 'device1' and len(e['R_sha256']) == 64

def test_device2_dataset_builds(fake_sensor_root, ds):
    assert DS.HFDSensorSim(fake_sensor_root, 'test', _cfg(), device='device2', setting=dict(mode='sequential', snr_db=30.0), prior_path=ds.prior_path)[0]['y'].shape == (16, 16, 30)

@pytest.mark.dataset
def test_x64_matches_the_legacy_loader(data_root):
    from data_loader.HFD_dataset import HFD_data
    from data_loader.warmstart_paths import PRIOR
    leg = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test', data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    new = DS.HFDSensorSim(data_root, 'test', dict(default_training_fields('p' * 64), P_y=50.0), setting=dict(mode='sequential', snr_db=60.0), prior_path=PRIOR)
    assert new.img_list == leg.img_list
    x_leg, y_leg = leg[3]; it = new[3]
    np.testing.assert_allclose(it['x'], x_leg, atol=1e-5)
    np.testing.assert_allclose(it['y'] - (y_leg + leg.sensor_R_matrix.sum(0)) / 2, 0, atol=6 * it['sigma'])   # physical y = (loader y + R^T 1) / 2 up to noise


def test_R_override_changes_y_and_sha(fake_sensor_root, ds):
    import hashlib
    R2 = 0.5 * ds.R
    o = DS.HFDSensorSim(ds.data_root, 'train', _cfg(), device='device1', prior_path=ds.prior_path,
                        setting=dict(mode='sequential', snr_db=60.0), R_override=R2)
    base = DS.HFDSensorSim(ds.data_root, 'train', _cfg(), device='device1', prior_path=ds.prior_path,
                           setting=dict(mode='sequential', snr_db=60.0))
    assert np.array_equal(o.R, R2) and o.sensor['sha256'] == base.sensor['sha256']
    assert o.checkpoint_extra()['R_sha256'] == hashlib.sha256(np.ascontiguousarray(R2, dtype=np.float64).tobytes()).hexdigest()
    assert o.checkpoint_extra()['R_sha256'] != base.checkpoint_extra()['R_sha256']
    assert not np.allclose(o[0]['y'], base[0]['y'])


# ----------------------------------------------------------------------------- Phase 3b scenarios
import hashlib  # noqa: E402

from data_loader import scenarios as SCN  # noqa: E402
from data_loader.mmse import mmse_maps, mmse_maps_general  # noqa: E402
from data_loader.sensor_forward import measure, sigma_from_snr  # noqa: E402
from data_loader.sensor_model import load_sensor  # noqa: E402

# sha256 over sorted (key, bytes) of item 3 on the fake root, computed with the code BEFORE the scenario work (a3a07cf):
# the family draw, sequential 30 dB, snapshot 20 dB (stable across OMP / MKL thread counts 1 and 4)
FROZEN = ('601261208effc963980407a3c98a4a07c497975bc91ea5a0b1b1308370cf7611',
          'b7a8ac8d3452c436a15b7574ee89291813acdc16810af5654e7afdcf23b66719',
          'c6980817138b99ef29407a96779168f04d1a7920925fe8f033397c02add4dbb0')
SETTINGS = (None, dict(mode='sequential', snr_db=30.0), dict(mode='snapshot', snr_db=20.0))


def _item_sha(it):
    m = hashlib.sha256()
    for k in sorted(it):
        m.update(k.encode()); m.update(np.ascontiguousarray(it[k]).tobytes())
    return m.hexdigest()


def _old_getitem(ds, idx):
    """The pre-scenario __getitem__, verbatim."""
    x01 = DS.load_cube01(ds.img_list[idx])
    mode, snr = ds.draw_setting(idx)
    sigma = sigma_from_snr(snr, ds.P_y)
    rng = np.random.default_rng([ds.config['seed'] + 1, ds.epoch, ds.file_key(idx)])
    y, mask = measure(x01, ds.R, mode, sigma, rng)
    snapshot = mode == 'snapshot'
    x0_hat, s64 = mmse_maps(y, mask if snapshot else None, ds.R, ds.prior, sigma, ds.A)
    x = ((2.0 * x01 - 1.0) @ ds.A.T).astype(np.float32)
    return dict(x=x, y=y, x0_hat=np.asarray(x0_hat, np.float32),
                s_post=np.maximum(s64, ds.s_floor_tr).astype(np.float32), s_raw=np.asarray(s64, np.float32),
                mask=mask, sigma=np.float32(sigma), snr_db=np.float32(snr), snapshot=np.float32(snapshot))


def test_no_scenario_items_are_bit_for_bit_unchanged(fake_sensor_root, ds):
    for st, sha in zip(SETTINGS, FROZEN):
        d = DS.HFDSensorSim(fake_sensor_root, 'train', _cfg(), device='device1', setting=st, prior_path=ds.prior_path)
        assert d.scenario is None and _item_sha(d[3]) == sha
        for e in (0, 2):
            d.set_epoch(e)
            for i in (0, 1, len(d) - 1):
                a, b = d[i], _old_getitem(d, i)
                assert set(a) == set(b) and all(a[k].tobytes() == np.asarray(b[k]).tobytes() for k in a)
    b = DS.sensor_collate_fn([ds[0], ds[1]])
    assert set(b) == set(DS.MAP_KEYS) | set(DS.SCALAR_KEYS)


def _scen(**kw):
    s = dict(name='t', R={'kind': 'device'}, states=None, pixels=None, noise={'kind': 'gaussian'})
    s.update(kw)
    return s


CONSTS = dict(mean_y=3.0, y_max=9.0, quantiles={'0.9': 5.0, '0.7': 4.0})
SEQ30 = dict(mode='sequential', snr_db=30.0)


def _sim(root, ds, scen, setting=SEQ30, **kw):
    return DS.HFDSensorSim(root, 'train', _cfg(), device=kw.pop('device', 'device1'), setting=setting,
                           prior_path=ds.prior_path, scenario=scen, scenario_consts=kw.pop('consts', CONSTS), **kw)


def test_scenario_R_measurement_and_statistics(fake_sensor_root, ds):
    sc = _scen(R={'kind': 'blend', 'alpha': 0.5}, states=10, noise={'kind': 'poisson'})
    d = _sim(fake_sensor_root, ds, sc)
    R = SCN.scenario_R(load_sensor('device1', fake_sensor_root)['R'], load_sensor('device2', fake_sensor_root)['R'], sc)
    assert np.array_equal(d.R, R) and d.R.shape == (31, 10) and d.A.shape == ds.A.shape
    assert d.state_indices.tolist() == [0, 3, 6, 10, 13, 16, 19, 23, 26, 29] and ds.state_indices is None
    assert d.R_sha256 == hashlib.sha256(np.ascontiguousarray(R, dtype=np.float64).tobytes()).hexdigest() != ds.R_sha256
    it = d[2]
    keys = {'x', 'y', 'x0_hat', 's_post', 's_raw', 'mask', 'sigma', 'snr_db', 'snapshot', 'noise_var',
            'y_naive', 'mask_naive', 'noise_var_naive', 'x0_hat_naive', 's_post_naive', 's_raw_naive'}
    assert set(it) == keys and it['y'].shape == it['noise_var'].shape == (16, 16, 10) and it['snapshot'] == 0
    sigma = sigma_from_snr(30.0, 50.0)
    rng = np.random.default_rng([_cfg()['seed'] + 1, 0, d.file_key(2)])
    m = SCN.measure_scenario(DS.load_cube01(d.img_list[2]), R, sigma, sc, rng, CONSTS)
    for k in ('y', 'mask', 'noise_var', 'y_naive', 'mask_naive', 'noise_var_naive'):
        assert np.array_equal(it[k], m[k]), k
    xh, s = mmse_maps_general(m['y'], m['mask'], m['noise_var'], R, d.prior, d.A)
    xn, sn = mmse_maps_general(m['y_naive'], m['mask_naive'], m['noise_var_naive'], R, d.prior, d.A)
    assert np.array_equal(it['x0_hat'], xh) and np.array_equal(it['s_raw'], s) and np.array_equal(it['s_post'], np.maximum(s, d.s_floor_tr))
    assert np.array_equal(it['x0_hat_naive'], xn) and np.array_equal(it['s_raw_naive'], sn)
    assert np.array_equal(it['s_post_naive'], np.maximum(sn, d.s_floor_tr))
    assert not np.array_equal(it['x0_hat'], it['x0_hat_naive'])
    assert np.array_equal(it['x'], ds[2]['x'])                                      # the truth does not depend on the scenario
    assert d.checkpoint_extra()['R_sha256'] == d.R_sha256


def test_pixel_scenario_returns_the_measured_flag_and_collates(fake_sensor_root, ds):
    sc = _scen(pixels={'kind': 'stride', 'd': 4})
    d = _sim(fake_sensor_root, ds, sc, consts=None)                                 # gaussian noise needs no constants
    it = d[0]
    assert set(it) == {'x', 'y', 'x0_hat', 's_post', 's_raw', 'mask', 'sigma', 'snr_db', 'snapshot', 'noise_var', 'measured'}
    st = SCN.pixel_measured(16, 16, sc['pixels'], None)
    assert it['measured'].shape == (16, 16, 1) and it['measured'].dtype == np.float32
    assert np.array_equal(it['measured'][..., 0], st.astype(np.float32)) and np.all(it['mask'][~st] == 0)
    b = DS.sensor_collate_fn([d[0], d[1]])
    assert b['measured'].shape == (2, 1, 16, 16) and b['noise_var'].shape == (2, 30, 16, 16) and 'y_naive' not in b
    dd = _sim(fake_sensor_root, ds, _scen(pixels={'kind': 'dead', 'p': 0.5}, noise={'kind': 'saturation', 'q': 0.9}))
    b = DS.sensor_collate_fn([dd[0], dd[1]])
    assert {'measured', 'noise_var', 'y_naive', 'mask_naive', 'noise_var_naive', 'x0_hat_naive', 's_post_naive',
            's_raw_naive'} <= set(b) and b['x0_hat_naive'].shape == (2, 64, 16, 16)
    assert not np.array_equal(dd[0]['measured'], dd[1]['measured'])               # the dead pixels are drawn per cube


def test_scenario_refusals(fake_sensor_root, ds):
    sc = _scen()
    with pytest.raises(ValueError, match='fixed'):
        _sim(fake_sensor_root, ds, sc, setting=None)
    with pytest.raises(ValueError, match='sequential'):
        _sim(fake_sensor_root, ds, sc, setting=dict(mode='snapshot', snr_db=30.0))
    with pytest.raises(ValueError, match='device1'):
        _sim(fake_sensor_root, ds, sc, device='device2')
    with pytest.raises(ValueError, match='R_override'):
        _sim(fake_sensor_root, ds, sc, R_override=ds.R)
    with pytest.raises(ValueError, match='constants'):
        _sim(fake_sensor_root, ds, _scen(noise={'kind': 'quantisation', 'bits': 4}), consts=None)
