import json
import os

from scripts.pilot.pilot_driver import next_actions, run_state

MODES = ('residual', 'standard', 'standard_xhat')


def mk_run(root, mode, epochs=(), done=None, alive=False, log_epochs=0):
    d = root / 'runs' / f'{mode}_d4_b64'; d.mkdir(parents=True, exist_ok=True)
    for e in epochs:
        (d / f'checkpoint_epoch_{e}.pth').write_bytes(b'x')
    if done is not None:
        (d / 'DONE').write_text(f'{done}\n')
    if alive:
        (d / 'pid').write_text(str(os.getpid()))             # this process: alive
    with open(d / 'train_log.jsonl', 'w') as f:
        for e in range(1, log_epochs + 1):
            f.write(json.dumps(dict(epoch=e, loss=1.0, lr=1e-4, sec=10, data_sec=1, elapsed_hours=0.1 * e, skipped_nonfinite=0)) + '\n')
    return str(d)


def mk_eval(root, mode, run, epoch, grid='full'):
    d = root / 'eval' / mode / 'd4'; d.mkdir(parents=True, exist_ok=True)
    (d / f'{run}__checkpoint_epoch_{epoch}{"" if grid == "full" else "__quick"}.json').write_text('{}')


def states(root):
    return {m: run_state(str(root / 'runs' / f'{m}_d4_b64'), str(root / 'eval')) for m in MODES}


def test_run_state_reads_markers(tmp_path):
    d = mk_run(tmp_path, 'residual', epochs=(10, 20, 25), alive=True, log_epochs=27)
    mk_eval(tmp_path, 'residual', 'residual_d4_b64', 25, 'quick')
    s = run_state(d, str(tmp_path / 'eval'))
    assert s['exists'] and s['alive'] and s['done'] is None and s['last_epoch'] == 27
    assert s['checkpoints'] == [10, 20, 25] and s['quick_evals'] == [25] and s['full_evals'] == []
    assert run_state(str(tmp_path / 'runs' / 'standard_d4_b64'), str(tmp_path / 'eval'))['exists'] is False


def test_initial_launch_assigns_b_and_a(tmp_path):
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert ('launch', 'residual', 0) in acts and ('launch', 'standard', 1) in acts and not any(a[1] == 'standard_xhat' for a in acts)


def test_dead_run_is_relaunched_and_busy_gpu_waits(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(10,), alive=False, log_epochs=12)
    mk_run(tmp_path, 'standard', alive=True, log_epochs=3)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'busy'})
    assert ('launch', 'residual', 0) in acts and not any(a[1] == 'standard' for a in acts)
    (tmp_path / 'runs' / 'residual_d4_b64' / 'relaunches').write_text('3\n')              # crash loop: left for a human
    assert not any(a[0] == 'launch' and a[1] == 'residual' for a in next_actions(states(tmp_path), {0: 'free', 1: 'busy'}))


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


def test_aprime_starts_when_a_done_and_full_evals_need_idle_gpu(tmp_path):
    mk_run(tmp_path, 'residual', epochs=(90, 100, 25, 50), alive=True, log_epochs=99)
    mk_run(tmp_path, 'standard', epochs=(90, 100, 25, 50), done=100, alive=False, log_epochs=100)
    acts = next_actions(states(tmp_path), {0: 'busy', 1: 'free'})
    assert ('launch', 'standard_xhat', 1) in acts
    assert not any(a[0] == 'eval_full' for a in acts)                 # GPU 1 goes to A′ first; A's full eval waits for an idle GPU
    mk_run(tmp_path, 'standard_xhat', alive=True, log_epochs=1)
    mk_run(tmp_path, 'residual', epochs=(90, 100, 25, 50), done=100, alive=False, log_epochs=100)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'busy'})
    assert acts and acts[0][:3] == ('eval_full', 'residual', 100) and acts[0][3] == 0 and len([a for a in acts if a[0] == 'eval_full']) == 1


def test_gate_when_three_full_evals_exist(tmp_path):
    for m in MODES:
        mk_run(tmp_path, m, epochs=(100,), done=100, log_epochs=100)
        mk_eval(tmp_path, m, f'{m}_d4_b64', 100)
    acts = next_actions(states(tmp_path), {0: 'free', 1: 'free'})
    assert acts == [('gate', 100, 100, 100)]
    (tmp_path / 'eval' / 'residual' / 'd4' / 'gate_1a__residual_d4_b64.json').write_text('{}')
    assert next_actions(states(tmp_path), {0: 'free', 1: 'free'}) == []
