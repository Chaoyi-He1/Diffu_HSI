import json, pytest
from data_loader import sensor_config as SC

def _cfg(**kw):
    c = SC.default_training_fields('p' * 64); c.update(P_y=50.0, devices={}, e2_demosaic=[0.0] * 30, n_files_P_y=10, created='t', prior_path='x'); c.update(kw); return c

def test_training_sha_ignores_non_training_keys_and_key_order():
    a, b = _cfg(created='t1', n_files_P_y=10), _cfg(created='t2', n_files_P_y=9000)
    assert SC.training_sha(a) == SC.training_sha(b)
    assert SC.training_sha(dict(reversed(list(a.items())))) == SC.training_sha(a)

def test_training_sha_changes_with_P_y_and_family():
    assert SC.training_sha(_cfg(P_y=50.0)) != SC.training_sha(_cfg(P_y=51.0))
    f = dict(_cfg()['family'], p_snapshot=0.4); assert SC.training_sha(_cfg(family=f)) != SC.training_sha(_cfg())

def test_check_config_refuses_missing_keys(tmp_path):
    c = _cfg(); del c['P_y']
    with pytest.raises(ValueError, match='P_y'):
        SC.check_config(c)
    p = tmp_path / 'c.json'; p.write_text(json.dumps(_cfg())); assert SC.load_config(str(p))['s_floor01'] == 0.01
    with pytest.raises(FileNotFoundError):
        SC.load_config(str(tmp_path / 'none.json'))

def test_make_config_refuses_overwrite(tmp_path, monkeypatch):
    from scripts.sensor import make_config as MC
    p = tmp_path / 'c.json'; p.write_text('{}')
    with pytest.raises(FileExistsError):
        MC.main(['--out', str(p), '--data_root', str(tmp_path), '--prior', 'none.npz'])

def test_make_config_refuses_limit_files_at_default_path(tmp_path, monkeypatch):
    from scripts.sensor import make_config as MC
    default = str(tmp_path / 'default.json')
    monkeypatch.setattr(MC, 'CONFIG_PATH', default)
    with pytest.raises(ValueError, match='--limit_files'):
        MC.main(['--out', default, '--limit_files', '4', '--data_root', str(tmp_path), '--prior', 'none.npz'])

def test_make_config_refuses_empty_file_list(tmp_path, monkeypatch):
    from scripts.sensor import make_config as MC
    import data_loader.HFD_dataset as HD
    class _Empty:
        def __init__(self, *a, **k): self.img_list = []
    monkeypatch.setattr(HD, 'HFD_data', _Empty)
    with pytest.raises(ValueError, match='data_root'):
        MC.main(['--out', str(tmp_path / 'o.json'), '--data_root', str(tmp_path), '--prior', 'none.npz'])
