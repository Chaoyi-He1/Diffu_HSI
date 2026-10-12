import fcntl
import json
import os
import subprocess
import time
import types

import pytest

from scripts import gate_1a
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


def test_run_cmd_returns_rc_and_kills_a_timed_out_group(monkeypatch):
    assert run_cmd(['true'], 30) == 0 and run_cmd(['false'], 30) == 1
    made = []

    class Recorded(subprocess.Popen):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)
    monkeypatch.setattr(pilot_driver.subprocess, 'Popen', Recorded)
    assert run_cmd(['bash', '-c', 'sleep 60 | cat'], 0.5) == 'timeout'                       # the whole pipeline is killed, not just bash
    pgid = made[0].pid                                                                        # its own session: pgid == pid
    deadline = time.monotonic() + 10                                                          # killed members may linger as zombies
    while time.monotonic() < deadline:                                                        # until init reaps them
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


def test_run_cmd_polls_and_calls_on_tick_between_polls(monkeypatch):
    assert pilot_driver.POLL_S == 60
    monkeypatch.setattr(pilot_driver, 'POLL_S', 0.2)
    ticks = []
    assert run_cmd(['bash', '-c', 'sleep 1; exit 3'], 30, on_tick=ticks.append) == 3       # rc passes through
    assert len(ticks) >= 3 and ticks == sorted(ticks) and all(0 < t < 30 for t in ticks)
    ticks = []
    assert run_cmd(['sleep', '60'], 1.0, on_tick=ticks.append) == 'timeout'               # the overall timeout still holds
    assert len(ticks) >= 2 and all(t < 1.0 for t in ticks)


def test_run_cmd_keeps_waiting_when_a_tick_raises(driver_root, monkeypatch):
    monkeypatch.setattr(pilot_driver, 'POLL_S', 0.2)

    def boom(elapsed):
        raise RuntimeError('tick failed')
    assert run_cmd(['bash', '-c', 'sleep 0.7'], 30, on_tick=boom) == 0                     # the child is waited for, not orphaned
    assert 'ERROR' in events(driver_root) and 'tick failed' in events(driver_root)


def test_a_launch_runs_while_a_synchronous_eval_is_in_flight(driver_root, monkeypatch):
    mk_run(driver_root, 'residual', epochs=(100,), done=100, log_epochs=100)              # B DONE: its full eval goes to GPU 0
    mk_run(driver_root, 'standard', epochs=(90,), alive=True, log_epochs=99)              # A still training on GPU 1
    gpus = {0: 'free', 1: 'busy'}
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: dict(gpus))
    calls = []

    def fake_run_cmd(cmd, timeout, on_tick=None):
        name, args = cmd[1].rsplit('/', 1)[-1], cmd[2:]
        calls.append(('start', name, args))
        if name == 'eval_milestone.sh':
            (driver_root / 'runs' / 'standard_d4_b64' / 'DONE').write_text('100\n')     # A finishes during B's eval ...
            gpus[1] = 'free'                                                              # ... and releases GPU 1
            (driver_root / 'runs' / 'standard_d4_b64' / 'checkpoint_epoch_50.pth').write_bytes(b'x')   # a quick eval is due: not in a tick
            on_tick(60.0)
            on_tick(120.0)                                                                # A' is alive now: not launched twice
        elif name == 'launch_run.sh':
            mk_run(driver_root, args[0], alive=True)                                      # the launcher wrote a live pid
        calls.append(('end', name, args))
        return 0
    monkeypatch.setattr(pilot_driver, 'run_cmd', fake_run_cmd)
    st, executed = pilot_driver.one_pass()
    assert [c[:2] for c in calls] == [('start', 'eval_milestone.sh'), ('start', 'launch_run.sh'), ('end', 'launch_run.sh'),
                                      ('end', 'eval_milestone.sh')]                       # launched before the eval returned
    assert calls[0][2] == ['residual', '100', 'full', '0'] and calls[1][2] == ['standard_xhat', '1']
    assert len([c for c in calls if c[1] == 'eval_milestone.sh']) == 2                     # one eval (start + end): none issued by a tick
    assert executed == [('eval_full', 'residual', 100, 0), ('launch', 'standard_xhat', 1)] and st['standard_xhat']['alive']
    log = events(driver_root)
    assert log.index("START ('launch', 'standard_xhat', 1)") < log.index("END   ('eval_full', 'residual', 100, 0)")


def test_the_in_flight_eval_gpu_counts_as_busy_in_a_tick(driver_root, monkeypatch):
    mk_run(driver_root, 'residual', epochs=(10,), alive=True, log_epochs=12, launches=1)
    mk_run(driver_root, 'standard', epochs=(100,), done=100, log_epochs=100)              # A DONE: its full eval goes to GPU 0
    mk_run(driver_root, 'standard_xhat', alive=True, log_epochs=3)
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: {0: 'free', 1: 'busy'})       # nvidia-smi does not list the eval yet
    calls = []

    def fake_run_cmd(cmd, timeout, on_tick=None):
        calls.append(cmd[1].rsplit('/', 1)[-1])
        if on_tick:
            os.remove(driver_root / 'runs' / 'residual_d4_b64' / 'pid')                   # B dies during the eval
            on_tick(60.0)
        return 0
    monkeypatch.setattr(pilot_driver, 'run_cmd', fake_run_cmd)
    st, executed = pilot_driver.one_pass()
    assert calls == ['eval_milestone.sh'] and executed == [('eval_full', 'standard', 100, 0)]   # no relaunch of B onto GPU 0 mid-eval
    assert ('launch', 'residual', 0) in next_actions(states(driver_root), {0: 'free', 1: 'busy'})  # the next pass relaunches it


def test_a_long_eval_logs_a_busy_heartbeat(driver_root, monkeypatch):
    assert pilot_driver.BUSY_EVERY_S == 600
    mk_run(driver_root, 'residual', epochs=(100,), done=100, log_epochs=100)
    mk_run(driver_root, 'standard', alive=True, log_epochs=5)
    monkeypatch.setattr(pilot_driver, 'gpu_status', lambda: {0: 'free', 1: 'busy'})
    monkeypatch.setattr(pilot_driver, 'BUSY_EVERY_S', 120)

    def fake_run_cmd(cmd, timeout, on_tick=None):
        for t in (60.0, 125.0, 180.0, 245.0, 300.0):
            on_tick(t)
        return 0
    monkeypatch.setattr(pilot_driver, 'run_cmd', fake_run_cmd)
    pilot_driver.one_pass()
    busy = [line for line in events(driver_root).splitlines() if ' BUSY ' in line]
    assert len(busy) == 2
    assert busy[0].endswith("BUSY ('eval_full', 'residual', 100, 0) elapsed=2min") and busy[1].endswith('elapsed=4min')


def test_a_launch_during_an_eval_leaves_the_rest_of_the_stale_plan_for_the_next_pass(driver_root, monkeypatch):
    plans = iter([[('eval_quick', 'residual', 25, 0), ('eval_full', 'standard', 100, 1)],   # the pass's plan
                  [('launch', 'standard_xhat', 1)]])                                        # the first tick: GPU 1 is taken now
    monkeypatch.setattr(pilot_driver, 'next_actions', lambda st, g: next(plans, []))
    ran = []

    def fake_execute(a, on_tick=None):
        ran.append(a)
        if on_tick:
            on_tick(60.0)
        return 0
    monkeypatch.setattr(pilot_driver, 'execute', fake_execute)
    st, executed = pilot_driver.one_pass()
    assert ran == executed == [('eval_quick', 'residual', 25, 0), ('launch', 'standard_xhat', 1)]   # no eval on the new run's GPU
    assert "REPLAN [('eval_full', 'standard', 100, 1)]" in events(driver_root)


def test_a_second_driver_exits_nonzero_while_the_lock_is_held(driver_root, monkeypatch):
    ran = []
    monkeypatch.setattr(pilot_driver, 'execute', lambda a, **k: ran.append(a) or 0)
    (driver_root / 'runs').mkdir(parents=True, exist_ok=True)
    with open(driver_root / 'runs' / 'pilot_driver.lock', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)                                  # another driver (--once or --loop) runs
        assert pilot_driver.main(['--once']) not in (0, None)
        assert pilot_driver.main(['--loop', '5']) not in (0, None)
        assert ran == []
    assert 'LOCKED' in events(driver_root)
    assert pilot_driver.main(['--once']) == 0 and ran                                     # free again: this driver runs ...
    assert pilot_driver.main(['--once']) == 0                                             # ... and released the lock on exit


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


def test_the_driver_gate_path_is_the_name_gate_1a_writes(tmp_path, monkeypatch):
    p = pilot_driver.gate_path(str(tmp_path / 'eval'))
    assert os.path.basename(p) == gate_1a.default_out_name(pilot_driver.run_name('residual')) == 'gate_1a__residual_d4_b64.json'
    assert p == str(tmp_path / 'eval' / 'residual' / 'd4' / 'gate_1a__residual_d4_b64.json')
    # the contract, not a copy of the string: if gate_1a renames its output, run_state follows
    monkeypatch.setattr(gate_1a, 'default_out_name', lambda run, *a, **k: f'renamed__{run}.json')
    (tmp_path / 'eval' / 'residual' / 'd4').mkdir(parents=True)
    assert states(tmp_path)['residual']['gate'] is False
    (tmp_path / 'eval' / 'residual' / 'd4' / 'renamed__residual_d4_b64.json').write_text('{}')
    assert states(tmp_path)['residual']['gate'] is True


def test_a_root_that_already_has_the_gate_file_is_refused(driver_root, monkeypatch):
    ran = []
    monkeypatch.setattr(pilot_driver, 'execute', lambda a, **k: ran.append(a) or 0)
    mk_gate(driver_root, '__nodone')                                                      # a peek does not make the root used ...
    assert pilot_driver.main(['--once']) == 0 and ran                                     # ... (control: this pass launches)
    ran.clear()
    mk_gate(driver_root)                                                                  # the official gate file: a used root
    assert pilot_driver.main(['--once']) not in (0, None)
    assert pilot_driver.main(['--loop', '5']) not in (0, None)
    assert ran == [] and 'RESIDUAL_WARMSTART_RESULTS' in events(driver_root)


LAUNCH = os.path.join(pilot_driver.REPO, 'scripts', 'pilot', 'launch_run.sh')


@pytest.fixture
def launcher(tmp_path):
    """launch_run.sh against a tmp results root, with stub nvidia-smi (GPU free) and python (records its args) on PATH."""
    stub = tmp_path / 'stub'
    stub.mkdir()
    (stub / 'nvidia-smi').write_text('#!/bin/bash\ncase "$*" in *memory.free*) echo 20000 ;; esac\n')
    (stub / 'python').write_text('#!/bin/bash\necho "$*" >> "$STUB_ARGS"\n')
    for f in ('nvidia-smi', 'python'):
        os.chmod(stub / f, 0o755)
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', RESIDUAL_WARMSTART_RESULTS=str(tmp_path / 'root'),
               PILOT_PY=str(stub / 'python'), STUB_ARGS=str(tmp_path / 'stub_args'))
    run = tmp_path / 'root' / 'runs' / 'residual_d4_b64'

    def launch():
        return subprocess.run(['bash', LAUNCH, 'residual', '0'], env=env, capture_output=True, text=True, timeout=60)
    return types.SimpleNamespace(launch=launch, run=run, args=tmp_path / 'stub_args')


def wait_for(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return path.exists()


def test_launch_run_starts_a_fresh_run_and_relaunches_a_dead_one(launcher):
    r = launcher.launch()
    assert r.returncode == 0, r.stdout + r.stderr
    assert (launcher.run / 'pid').read_text().strip().isdigit() and (launcher.run / 'launches').read_text().strip() == '1'
    assert wait_for(launcher.args) and '--mode residual' in launcher.args.read_text() and '--max_hours 24' in launcher.args.read_text()
    p = subprocess.Popen(['true']); p.wait()
    (launcher.run / 'pid').write_text(f'{p.pid}\n')                                       # a stale pid: the process is gone
    r = launcher.launch()
    assert r.returncode == 0 and (launcher.run / 'launches').read_text().strip() == '2'


def test_launch_run_refuses_a_done_run(launcher):
    launcher.run.mkdir(parents=True)
    (launcher.run / 'DONE').write_text('100\n')
    r = launcher.launch()
    assert r.returncode != 0 and 'DONE' in r.stdout
    assert not (launcher.run / 'pid').exists() and not (launcher.run / 'launches').exists() and not launcher.args.exists()


def test_launch_run_refuses_a_run_whose_pid_is_alive(launcher):
    launcher.run.mkdir(parents=True)
    (launcher.run / 'pid').write_text(f'{os.getpid()}\n')                                 # this process: alive
    r = launcher.launch()
    assert r.returncode != 0
    assert (launcher.run / 'pid').read_text().strip() == str(os.getpid())                 # no new pid
    assert not (launcher.run / 'launches').exists() and not launcher.args.exists()


def test_launch_run_refuses_while_another_launcher_holds_the_lock(launcher):
    launcher.run.mkdir(parents=True)
    with open(launcher.run / '.launch.lock', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = launcher.launch()
    assert r.returncode != 0
    assert not (launcher.run / 'pid').exists() and not (launcher.run / 'launches').exists() and not launcher.args.exists()
    assert launcher.launch().returncode == 0                                              # released: the next launch goes ahead
