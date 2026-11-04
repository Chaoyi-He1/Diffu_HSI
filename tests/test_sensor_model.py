import numpy as np, pytest, scipy.io as sio
from data_loader import sensor_model as SM

def test_device2_file_is_transposed_before_use(fake_sensor_root):
    assert SM.raw_matrix('device2', fake_sensor_root).shape == (400, 42)
    assert SM.raw_matrix('device1', fake_sensor_root).shape == (401, 71)

def test_legacy_axes():
    np.testing.assert_allclose(SM.legacy_axis('device1', 401)[[0, -1]], [400, 1000])
    assert len(SM.legacy_axis('ph5', 116)) == 116 and SM.legacy_axis('ph5', 116)[0] == 400
    with pytest.raises(ValueError):
        SM.legacy_axis('ph5', 100)

def test_load_sensor_shapes_selection_and_normalisation(fake_sensor_root):
    s = SM.load_sensor('device1', fake_sensor_root)
    assert s['R'].shape == (31, 30) and s['R_raw'].shape == (31, 30)
    assert s['selected_states'] == np.linspace(0, 70, 30, dtype=int).tolist() and s['n_states_total'] == 71
    np.testing.assert_allclose(s['R'].min(0), 0, atol=1e-12); np.testing.assert_allclose(s['R'].max(0), 1, atol=1e-12)
    assert s['axis_rule'] == 'legacy' and len(s['sha256']) == 64
    assert SM.load_sensor('ph5', fake_sensor_root)['R'].shape == (31, 30)

def test_sha_changes_with_device(fake_sensor_root):
    assert SM.load_sensor('device1', fake_sensor_root)['sha256'] != SM.load_sensor('device2', fake_sensor_root)['sha256']

@pytest.mark.dataset
def test_device1_equals_legacy_loader(data_root):
    from data_loader.HFD_dataset import HFD_data
    ds = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test', data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    np.testing.assert_allclose(SM.load_sensor('device1', data_root)['R'], ds.sensor_R_matrix, atol=1e-12)
