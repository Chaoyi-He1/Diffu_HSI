import json
import os
import types

import pytest

from scripts.pilot import pilot_driver
from scripts.pilot.pilot_driver import (MAX_ACTION_FAILURES, executable, is_stuck, load_failures, next_actions,
                                        parse_gpu_status, run_cmd, run_state)

MODES = ('residual', 'standard', 'standard_xhat')
real_gpu_status = pilot_driver.gpu_status


def mk_run(root, mode, epochs=(), done=None, alive=False, log_epochs=0, launches=None):
    d = root / 'runs' / f'{mode}_d4_b64'; d.mkdir(parents=True, exist_ok=True)
    for e in epochs:
        (d / f'checkpoint_epoch_{e}.pth').write_bytes(b'x')
    if done is not None:
        (d / 'DONE').write_text(f'{done}\n')
    if alive:
        (d / 'pid').write_text(str(os.getpid()))             # this process: alive
    if launches is not None:
        (d / 'launches').write_text(f'{launches}\n')
    with open(d / 'train_log.jsonl', 'w') as f:
        for e in range(1, log_epochs + 1):
            f.write(json.dumps(dict(epoch=e, loss=1.0, lr=1e-4, sec=10, data_sec=1, elapsed_hours=0.1 * e, skipped_nonfinite=0)) + '\n')
    return str(d)


def mk_eval(root, mode, run, epoch, grid='full'):
    d = root / 'eval' / mode / 'd4'; d.mkdir(parents=True, exist_ok=True)
    (d / f'{run}__checkpoint_epoch_{epoch}{"" if grid == "full" else "__quick"}.json').write_text('{}')


def mk_gate(root, suffix=''):
    d = root / 'eval' / 'residual' / 'd4'; d.mkdir(parents=True, exist_ok=True)
    (d / f'gate_1a__residual_d4_b64{suffix}.json').write_text('{}')


def states(root):
    return {m: run_state(str(root / 'runs' / f'{m}_d4_b64'), str(root / 'eval')) for m in MODES}


def events(root):
    p = root / 'runs' / 'pilot_events.log'
    return p.read_text() if p.exists() else ''


@pytest.fixture
def driver_root(tmp_path, monkeypatch):
    """The driver pointed at a tmp results root, with no nvidia-smi and no sleeping."""
    monkeypatch.setattr(pilot_driver, 'ROOT', str(tmp_path))
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: {0: 'free', 1: 'free'})
    monkeypatch.setattr(pilot_driver, 'time', types.SimpleNamespace(sleep=lambda s: pytest.fail('the loop must not sleep here')))
    return tmp_path


def test_run_state_reads_markers(tmp_path):
    d = mk_run(tmp_path, 'residual', epochs=(10, 20, 25), alive=True, log_epochs=27)
    mk_eval(tmp_path, 'residual', 'residual_d4_b64', 25, 'quick')
    s = run_state(d, str(tmp_path / 'eval'))
    assert s['exists'] and s['alive'] and s['done'] is None and s['last_epoch'] == 27
    assert s['checkpoints'] == [10, 20, 25] and s['quick_evals'] == [25] and s['full_evals'] == []
    assert run_state(str(tmp_path / 'runs' / 'standard_d4_b64'), str(tmp_path / 'eval'))['exists'] is False


def test_run_state_is_lenient_about_half_written_markers(tmp_path):
    d = mk_run(tmp_path, 'residual', epochs=(10,), log_epochs=3, launches=2)
    (tmp_path / 'runs' / 'residual_d4_b64' / 'DONE').write_text('')                      # the writer has opened but not written yet
    assert run_state(d, str(tmp_path / 'eval'))['done'] is None
    (tmp_path / 'runs' / 'residual_d4_b64' / 'DONE').write_text('not a number\n')
    assert run_state(d, str(tmp_path / 'eval'))['done'] is None
    (tmp_path / 'runs' / 'residual_d4_b64' / 'launches').write_text('garbage\n')
    (tmp_path / 'runs' / 'residual_d4_b64' / 'pid').write_text('\n')
    s = run_state(d, str(tmp_path / 'eval'))
    assert s['launches'] == 0 and s['alive'] is False
    with open(tmp_path / 'runs' / 'residual_d4_b64' / 'train_log.jsonl', 'a') as f:
        f.write('{"epoch": 4, "lo')                                                       # half-written last line
    assert run_state(d, str(tmp_path / 'eval'))['last_epoch'] == 3


def test_initial_launch_assigns_b_and_a(tmp_path):
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert ('launch', 'residual', 0) in acts and ('launch', 'standard', 1) in acts and not any(a[1] == 'standard_xhat' for a in acts)


def test_dead_run_is_relaunched_and_busy_gpu_waits(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(10,), alive=False, log_epochs=12)
    mk_run(tmp_path, 'standard', alive=True, log_epochs=3)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'busy'})
    assert ('launch', 'residual', 0) in acts and not any(a[1] == 'standard' for a in acts)
    (tmp_path / 'runs' / 'residual_d4_b64' / 'launches').write_text('4\n')                 # crash loop: left for a human
    assert not any(a[0] == 'launch' and a[1] == 'residual' for a in next_actions(states(tmp_path), {0: 'free', 1: 'busy'}))


def test_launch_cap_counts_every_launch_even_before_the_first_checkpoint(tmp_path):
    mk_run(tmp_path, 'residual', alive=False, log_epochs=2, launches=3)                    # no checkpoint: 1 + 3 launches allowed
    assert ('launch', 'residual', 0) in next_actions(states(tmp_path), {0: 'free', 1: 'busy'})
    (tmp_path / 'runs' / 'residual_d4_b64' / 'launches').write_text('4\n')
    assert not any(a[0] == 'launch' and a[1] == 'residual' for a in next_actions(states(tmp_path), {0: 'free', 1: 'free'}))


def test_busy_gpu_blocks_the_launch_of_a_dead_run(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(10,), alive=False, log_epochs=12)                 # dead and not DONE, but GPU 0 is taken
    mk_run(tmp_path, 'standard', alive=True, log_epochs=3)
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'busy'})
    assert not any(a[0] == 'launch' for a in acts)


def test_run_state_mode_parsing_for_other_run_names(tmp_path):
    d = tmp_path / 'runs' / 'residual_d4_b256'; d.mkdir(parents=True)
    (tmp_path / 'eval' / 'residual' / 'd4').mkdir(parents=True)
    (tmp_path / 'eval' / 'residual' / 'd4' / 'residual_d4_b256__checkpoint_epoch_200.json').write_text('{}')
    s = run_state(str(d), str(tmp_path / 'eval'))
    assert s['mode'] == 'residual' and s['full_evals'] == [200]


def test_quick_eval_when_milestone_present_and_json_absent(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(20, 25), alive=True, log_epochs=26)
    mk_run(tmp_path, 'standard', epochs=(20, 25), alive=True, log_epochs=26)
    mk_eval(tmp_path, 'standard', 'standard_d4_b64', 25, 'quick')
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'busy'})
    assert ('eval_quick', 'residual', 25, 0) in acts and not any(a[0] == 'eval_quick' and a[1] == 'standard' for a in acts)


def test_epoch_50_quick_eval_and_missed_quick_evals_still_run_after_done(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(25, 50), alive=True, log_epochs=51)
    mk_eval(tmp_path, 'residual', 'residual_d4_b64', 25, 'quick')
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'busy'})
    assert [a for a in acts if a[0] == 'eval_quick'] == [('eval_quick', 'residual', 50, 0)]
    mk_run(tmp_path, 'standard', epochs=(25, 50, 100), done=100, log_epochs=100)           # DONE, but its epoch-50 quick eval was missed
    mk_eval(tmp_path, 'standard', 'standard_d4_b64', 25, 'quick')
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'busy'})
    assert ('eval_quick', 'standard', 50, 1) in acts and not any(a[0] == 'eval_quick' and a[2] == 25 for a in acts)


def test_aprime_starts_when_a_done_and_full_evals_need_idle_gpu(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(90, 100, 25, 50), alive=True, log_epochs=99)
    mk_run(tmp_path, 'standard', epochs=(90, 100, 25, 50), done=100, alive=False, log_epochs=100)
    for m in ('residual', 'standard'):
        for e in (25, 50):
            mk_eval(tmp_path, m, f'{m}_d4_b64', e, 'quick')
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'free'})
    assert ('launch', 'standard_xhat', 1) in acts
    assert not any(a[0] == 'eval_full' for a in acts)                 # GPU 1 goes to A′ first; A's full eval waits for an idle GPU
    mk_run(tmp_path, 'standard_xhat', alive=True, log_epochs=1)
    mk_run(tmp_path, 'residual', epochs=(90, 100, 25, 50), done=100, alive=False, log_epochs=100)
    os.remove(tmp_path / 'runs' / 'residual_d4_b64' / 'pid')                               # B has exited (mk_run keeps an old pid)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'busy'})
    assert acts and acts[0][:3] == ('eval_full', 'residual', 100) and acts[0][3] == 0 and len([a for a in acts if a[0] == 'eval_full']) == 1


def test_aprime_waits_while_a_is_alive(tmp_path):
    mk_run(tmp_path, 'residual', alive=True, log_epochs=5)
    mk_run(tmp_path, 'standard', alive=True, log_epochs=5)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert not any(a[1] == 'standard_xhat' for a in acts)


def test_two_idle_gpus_take_two_full_evals_on_distinct_gpus(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(100,), done=100, log_epochs=100)
    mk_run(tmp_path, 'standard', epochs=(100,), done=100, log_epochs=100)
    mk_run(tmp_path, 'standard_xhat', epochs=(100,), done=100, log_epochs=100)
    mk_eval(tmp_path, 'standard_xhat', 'standard_xhat_d4_b64', 100)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert acts == [('eval_full', 'residual', 100, 0), ('eval_full', 'standard', 100, 1)]


def test_gate_when_three_full_evals_exist(tmp_path):
    for m in MODES:
        mk_run(tmp_path, m, epochs=(100,), done=100, log_epochs=100)
        mk_eval(tmp_path, m, f'{m}_d4_b64', 100)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert acts == [('gate', 100, 100, 100)]
    mk_gate(tmp_path, '__nodone')                                     # a --no_require_done peek is not the official gate
    assert states(tmp_path)['residual']['gate'] is False
    assert next_actions(states(tmp_path), {0: 'free', 1: 'free'}) == [('gate', 100, 100, 100)]
    mk_gate(tmp_path)
    assert states(tmp_path)['residual']['gate'] is True
    assert next_actions(states(tmp_path), {0: 'free', 1: 'free'}) == []


def test_an_alive_run_with_done_gets_neither_full_eval_nor_gate(tmp_path):
    # B has written DONE but its process is still alive (exiting, or a manual resume): no full eval of it yet
    mk_run(tmp_path, 'residual', epochs=(100,), done=100, alive=True, log_epochs=100)
    mk_run(tmp_path, 'standard', epochs=(90,), alive=True, log_epochs=95)
    assert not any(a[0] in ('eval_full', 'gate') for a in next_actions(states(tmp_path), {0: 'free', 1: 'free'}))
    os.remove(tmp_path / 'runs' / 'residual_d4_b64' / 'pid')                               # control: once it is gone, it is evaluated
    assert ('eval_full', 'residual', 100, 0) in next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    # all three DONE and evaluated, but A' is alive again (resumed past an old DONE): no gate
    root = tmp_path / 'all_done'
    for m in MODES:
        mk_run(root, m, epochs=(100,), done=100, log_epochs=100)
        mk_eval(root, m, f'{m}_d4_b64', 100)
    (root / 'runs' / 'standard_xhat_d4_b64' / 'pid').write_text(str(os.getpid()))
    assert not any(a[0] in ('eval_full', 'gate') for a in next_actions(states(root), {0: 'free', 1: 'free'}))
    os.remove(root / 'runs' / 'standard_xhat_d4_b64' / 'pid')
    assert next_actions(states(root), {0: 'free', 1: 'free'}) == [('gate', 100, 100, 100)]


def test_gate_waits_for_the_third_full_eval(tmp_path):
    for m in MODES:
        mk_run(tmp_path, m, epochs=(100,), done=100, log_epochs=100)
    for m in ('residual', 'standard'):
        mk_eval(tmp_path, m, f'{m}_d4_b64', 100)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert acts == [('eval_full', 'standard_xhat', 100, 0)]


def test_failure_bookkeeping_helpers(tmp_path):
    p = str(tmp_path / 'action_failures.json')
    assert load_failures(p) == {}                                                           # absent
    (tmp_path / 'action_failures.json').write_text('not json')
    assert load_failures(p) == {}                                                           # corrupt
    (tmp_path / 'action_failures.json').write_text('[1, 2]')
    assert load_failures(p) == {}                                                           # wrong shape
    a, b = ('eval_full', 'residual', 100, 0), ('launch', 'standard', 1)
    (tmp_path / 'action_failures.json').write_text(json.dumps({repr(a): MAX_ACTION_FAILURES, repr(b): MAX_ACTION_FAILURES - 1}))
    f = load_failures(p)
    assert f == {repr(a): MAX_ACTION_FAILURES, repr(b): MAX_ACTION_FAILURES - 1}
    assert executable([a, b, ('gate', 1, 2, 3)], f) == [b, ('gate', 1, 2, 3)]


def test_execute_records_failures_timeouts_and_clears_on_success(driver_root, monkeypatch):
    calls, rcs = [], iter([1, 'timeout', 0])
    monkeypatch.setattr(pilot_driver, 'run_cmd', lambda cmd, timeout: (calls.append((cmd, timeout)), next(rcs))[1])
    a = ('eval_full', 'residual', 100, 1)
    path = pilot_driver.failures_path()
    assert pilot_driver.execute(a) == 1 and load_failures(path) == {repr(a): 1}
    assert pilot_driver.execute(a) == 'timeout' and load_failures(path) == {repr(a): 2}
    assert executable([a], load_failures(path)) == []
    assert pilot_driver.execute(a) == 0 and load_failures(path) == {}
    cmd, timeout = calls[0]
    assert cmd[-4:] == ['residual', '100', 'full', '1'] and cmd[1].endswith('eval_milestone.sh') and timeout == 8 * 3600
    assert 'END   ' + repr(a) + ' rc=timeout' in events(driver_root)


def test_execute_timeouts_per_action_kind(driver_root, monkeypatch):
    seen = []
    monkeypatch.setattr(pilot_driver, 'run_cmd', lambda cmd, timeout: seen.append((cmd[1].rsplit('/', 1)[-1], timeout)) or 0)
    for a in (('launch', 'residual', 0), ('eval_quick', 'residual', 25, 0), ('eval_full', 'residual', 100, 0), ('gate', 100, 100, 100)):
        pilot_driver.execute(a)
    assert seen == [('launch_run.sh', 3600), ('eval_milestone.sh', 2 * 3600), ('eval_milestone.sh', 8 * 3600), ('gate_1a.py', 3600)]


def test_run_cmd_returns_rc_and_kills_a_timed_out_group():
    assert run_cmd(['true'], 30) == 0 and run_cmd(['false'], 30) == 1
    assert run_cmd(['bash', '-c', 'sleep 60 | cat'], 0.5) == 'timeout'                       # the whole pipeline is killed, not just bash


def test_gpu_status_parsing_skips_unparsable_lines():
    uuids = 'garbage\n0, GPU-aaa\n1, GPU-bbb\n\n,,,\nx, GPU-ccc\n'
    apps = 'No devices were found\nGPU-bbb, 1234\n\nGPU-unknown, 5\n'
    assert parse_gpu_status(uuids, apps) == {0: 'free', 1: 'busy'}


def test_gpu_status_logs_once_per_nvidia_smi_failure_episode(driver_root, monkeypatch):
    monkeypatch.setattr(pilot_driver, '_SMI_FAILING', False)
    monkeypatch.setattr(pilot_driver, '_smi', lambda args: None)
    assert real_gpu_status() == {} and real_gpu_status() == {}
    assert events(driver_root).count('nvidia-smi') == 1


def test_is_stuck_means_nothing_executable_and_nothing_alive(tmp_path):
    mk_run(tmp_path, 'residual', alive=False, log_epochs=2, launches=4)
    assert is_stuck(states(tmp_path), [])
    assert not is_stuck(states(tmp_path), [('launch', 'standard', 1)])
    mk_run(tmp_path, 'standard', alive=True, log_epochs=2)
    assert not is_stuck(states(tmp_path), [])


def test_is_stuck_never_on_a_failed_nvidia_smi(tmp_path):
    mk_run(tmp_path, 'residual', alive=False, log_epochs=2, launches=4)
    assert not is_stuck(states(tmp_path), [], {})                                         # nvidia-smi failed: wait
    assert is_stuck(states(tmp_path), [], {0: 'busy', 1: 'busy'})                         # foreign jobs hold the GPUs: STUCK


def test_a_failing_nvidia_smi_makes_the_loop_wait_not_stuck(driver_root, monkeypatch):
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: {})                           # nvidia-smi fails on every pass
    slept = []

    class Stop(Exception):
        pass

    def sleep(s):
        slept.append(s)
        if len(slept) == 2:
            raise Stop
    monkeypatch.setattr(pilot_driver, 'time', types.SimpleNamespace(sleep=sleep))
    with pytest.raises(Stop):                                                             # rc 2 before the amendment
        pilot_driver.main(['--loop', '5'])
    log = events(driver_root)
    assert slept == [5, 5] and log.count('SMI-FAIL') == 2 and 'STUCK' not in log


def test_gpus_held_by_foreign_jobs_with_our_runs_dead_is_still_stuck(driver_root, monkeypatch):
    mk_run(driver_root, 'residual', epochs=(10,), log_epochs=12, launches=1)               # dead and relaunchable, but GPU 0 is taken
    mk_run(driver_root, 'standard', log_epochs=2, launches=1)
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: {0: 'busy', 1: 'busy'})
    assert pilot_driver.main(['--loop', '5']) == 2
    log = events(driver_root)
    assert 'STUCK' in log and 'SMI-FAIL' not in log


def test_one_pass_survives_an_exception_and_logs_it(driver_root, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('half-written marker')
    monkeypatch.setattr(pilot_driver, 'run_state', boom)
    assert pilot_driver.one_pass() == (None, [])                                          # does not propagate
    log = events(driver_root)
    assert 'ERROR' in log and 'half-written marker' in log


def test_one_pass_logs_idle_heartbeat_and_skips_capped_actions(driver_root, monkeypatch):
    mk_run(driver_root, 'residual', alive=True, log_epochs=3)
    mk_run(driver_root, 'standard', alive=True, log_epochs=3)
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: {0: 'busy', 1: 'busy'})
    st, todo = pilot_driver.one_pass()
    assert todo == [] and st['residual']['alive']
    assert "IDLE alive=['residual', 'standard']" in events(driver_root) and "'standard_xhat': None" in events(driver_root)
    # a launch whose failure count has reached the cap is dropped (and logged), the rest still run
    mk_run(driver_root, 'residual', epochs=(10,), alive=False, log_epochs=12)
    os.remove(driver_root / 'runs' / 'residual_d4_b64' / 'pid')                            # now dead
    (driver_root / 'runs' / 'action_failures.json').write_text(json.dumps({repr(('launch', 'residual', 0)): MAX_ACTION_FAILURES}))
    ran = []
    monkeypatch.setattr(pilot_driver, 'execute', lambda a: ran.append(a) or 0)
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: {0: 'free', 1: 'busy'})
    st, todo = pilot_driver.one_pass()
    assert ran == [] and todo == []
    assert f"SKIP ('launch', 'residual', 0) (failed {MAX_ACTION_FAILURES} times)" in events(driver_root)


def test_loop_exits_with_stuck_when_nothing_can_progress(driver_root, monkeypatch):
    mk_run(driver_root, 'residual', epochs=(10,), alive=False, log_epochs=12, launches=4)
    mk_run(driver_root, 'standard', alive=False, log_epochs=2, launches=4)
    assert pilot_driver.main(['--loop', '5']) == 2                                         # nothing to launch, nothing alive: no sleep
    assert 'STUCK' in events(driver_root)


def test_loop_exits_in_the_same_pass_once_the_gate_has_run(driver_root, monkeypatch):
    for m in MODES:
        mk_run(driver_root, m, epochs=(100,), done=100, log_epochs=100)
        mk_eval(driver_root, m, f'{m}_d4_b64', 100)
    ran = []
    def fake_run_cmd(cmd, timeout):
        ran.append(cmd); mk_gate(driver_root); return 0
    monkeypatch.setattr(pilot_driver, 'run_cmd', fake_run_cmd)
    assert pilot_driver.main(['--loop', '5']) == 0                                         # sleep() would fail the test
    assert len(ran) == 1 and ran[0][1].endswith('gate_1a.py') and ran[0][-1].endswith('standard_xhat_d4_b64__checkpoint_epoch_100.json')
    log = events(driver_root)
    assert "START ('gate', 100, 100, 100)" in log and 'rc=0' in log


def test_a_failing_gate_is_retried_once_then_the_loop_reports_stuck(driver_root, monkeypatch):
    for m in MODES:
        mk_run(driver_root, m, epochs=(100,), done=100, log_epochs=100)
        mk_eval(driver_root, m, f'{m}_d4_b64', 100)
    ran = []
    monkeypatch.setattr(pilot_driver, 'run_cmd', lambda cmd, timeout: ran.append(cmd) or 1)
    monkeypatch.setattr(pilot_driver, 'time', types.SimpleNamespace(sleep=lambda s: None))
    assert pilot_driver.main(['--loop', '5']) == 2
    assert len(ran) == MAX_ACTION_FAILURES and 'STUCK' in events(driver_root) and 'SKIP' in events(driver_root)
