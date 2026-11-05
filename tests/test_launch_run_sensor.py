"""launch_run.sh knows the four sensor modes (Phase 3 Task 8): --sensor_config is required, --ds 1 is appended after the
caller's arguments (an explicit other --ds is refused), the regress guards cover regress_fam and varhead_fam needs
--loss_type nll, and the longer-mode RUN_NAME guard covers regress_fam_* / varhead_fam_*."""
import pytest

from test_launch_run_name import env, launch, stub_lines, wait_dead  # noqa: F401  (env is a fixture)

CFG = ('--sensor_config', '/c.json')
DIRECT = ('--prediction_type', 'direct')


def _nothing_started(tmp_path):
    return not (tmp_path / 'root').exists() and not (tmp_path / 'stub_args').exists()


def test_sensor_mode_without_config_is_refused(env, tmp_path):
    r = launch(env, 'uncond_b64', 'uncond', 0)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and 'needs --sensor_config' in out, out
    assert _nothing_started(tmp_path)


def test_sensor_mode_appends_ds_1_and_passes_config(env, tmp_path):
    r = launch(env, 'cond_fam_b64', 'cond_fam', 0, *CFG)
    assert r.returncode == 0, r.stdout + r.stderr
    a = stub_lines(tmp_path, 1)[0]
    assert '--mode cond_fam' in a and '--ds 4 ' in a and a.endswith('--sensor_config /c.json --ds 1')
    wait_dead(tmp_path / 'root' / 'runs' / 'cond_fam_b64')
    r = launch(env, 'uncond_b64', 'uncond', 0, '--sensor_config=/c.json')
    assert r.returncode == 0, r.stdout + r.stderr
    assert stub_lines(tmp_path, 2)[1].endswith('--sensor_config=/c.json --ds 1')


@pytest.mark.parametrize('ds', [('--ds', '4'), ('--ds=4',), ('--sensor_down_sample_rate', '2'), ('--ds', '1', '--ds', '8')])
def test_sensor_mode_refuses_other_ds(env, tmp_path, ds):
    r = launch(env, 'uncond_b64', 'uncond', 0, *CFG, *ds)
    assert r.returncode == 1 and 'not launching' in r.stdout, r.stdout + r.stderr
    assert _nothing_started(tmp_path)


def test_sensor_mode_accepts_an_explicit_ds_1(env, tmp_path):
    r = launch(env, 'uncond_b64', 'uncond', 0, *CFG, '--ds', '1')
    assert r.returncode == 0, r.stdout + r.stderr
    assert stub_lines(tmp_path, 1)[0].endswith('--sensor_config /c.json --ds 1 --ds 1')


@pytest.mark.parametrize('run_name, extra, why', [
    ('regress_fam_b64', CFG + DIRECT, 'needs --loss_type'),
    ('regress_fam_b64', CFG + DIRECT + ('--loss_type', 'nll'), 'needs --loss_type'),
    ('regress_fam_l2_b64', CFG + DIRECT + ('--loss_type', 'l1'), 'names l2 but --loss_type is l1'),
    ('regress_fam_l1_s43_b64', CFG + DIRECT + ('--loss_type', 'l1'), "names seed 43 but --seed is ''"),
])
def test_regress_fam_needs_loss_and_direct_like_regress(env, tmp_path, run_name, extra, why):
    r = launch(env, run_name, 'regress_fam', 0, *extra)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and why in out, out
    assert _nothing_started(tmp_path)


def test_regress_fam_is_accepted_with_a_matching_loss(env, tmp_path):
    r = launch(env, 'regress_fam_l2_b64', 'regress_fam', 1, *CFG, *DIRECT, '--loss_type', 'l2')
    assert r.returncode == 0, r.stdout + r.stderr
    a = stub_lines(tmp_path, 1)[0]
    assert '--mode regress_fam' in a and a.endswith('--loss_type l2 --ds 1')


@pytest.mark.parametrize('loss', [(), ('--loss_type', 'l1')])
def test_varhead_fam_needs_nll(env, tmp_path, loss):
    r = launch(env, 'varhead_fam_b64', 'varhead_fam', 0, *CFG, *DIRECT, *loss)
    assert r.returncode == 1 and 'needs --loss_type nll' in r.stdout, r.stdout + r.stderr
    assert _nothing_started(tmp_path)


def test_varhead_fam_is_accepted_with_nll(env, tmp_path):
    r = launch(env, 'varhead_fam_b64', 'varhead_fam', 0, *CFG, *DIRECT, '--loss_type', 'nll')
    assert r.returncode == 0, r.stdout + r.stderr
    assert stub_lines(tmp_path, 1)[0].endswith('--loss_type nll --ds 1')


@pytest.mark.parametrize('run_name, mode, extra, longer', [
    ('regress_fam_b64', 'regress', DIRECT + ('--loss_type', 'l1'), 'regress_fam'),
    ('varhead_fam_b64', 'varhead', DIRECT + ('--loss_type', 'nll'), 'varhead_fam'),
])
def test_longer_mode_name_guard_covers_fam(env, tmp_path, run_name, mode, extra, longer):
    r = launch(env, run_name, mode, 0, *extra)
    assert r.returncode == 1 and f'belongs to mode {longer}, not {mode}' in r.stdout, r.stdout + r.stderr
    assert _nothing_started(tmp_path)


def test_uncond_tt_is_a_sensor_mode(env, tmp_path):
    r = launch(env, 'uncond_tt_b64', 'uncond_tt', 0)
    assert r.returncode == 1 and 'uncond_tt needs --sensor_config' in r.stdout, r.stdout + r.stderr
    assert _nothing_started(tmp_path)
    r = launch(env, 'uncond_tt_b64', 'uncond_tt', 0, *CFG)
    assert r.returncode == 0, r.stdout + r.stderr
    a = stub_lines(tmp_path, 1)[0]
    assert '--mode uncond_tt' in a and a.endswith('--sensor_config /c.json --ds 1')
    assert (tmp_path / 'root' / 'runs' / 'uncond_tt_b64').is_dir()


def test_longer_mode_name_guard_covers_uncond_tt(env, tmp_path):
    r = launch(env, 'uncond_tt_x', 'uncond', 0, *CFG)
    assert r.returncode == 1 and 'belongs to mode uncond_tt, not uncond' in r.stdout, r.stdout + r.stderr
    assert _nothing_started(tmp_path)
