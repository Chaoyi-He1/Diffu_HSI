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
    draws = [ds.draw_setting(i) for i in range(2000)]
    assert abs(np.mean([m == 'snapshot' for m, _ in draws]) - 0.5) < 0.04
    assert abs(np.mean([s == 60.0 for _, s in draws]) - 0.1) < 0.03
    others = [s for _, s in draws if s != 60.0]; assert min(others) >= 20.0 and max(others) <= 50.0

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
