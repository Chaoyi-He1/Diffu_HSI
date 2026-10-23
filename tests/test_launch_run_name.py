"""launch_run.sh honours RUN_NAME (Phase 1a-prime Task 5): the run directory is RUN_NAME and the extra arguments reach
the trainer last; a relaunch resumes from that directory's own checkpoints; DONE, pid, launches and run.log are per run;
an empty RUN_NAME falls back to <mode>_d4_b64. A malformed RUN_NAME, and a regress launch without a --loss_type that
matches the name, are refused before anything is created or started."""
import os
import subprocess
import time

import pytest

from data_loader.warmstart_paths import REPO

LAUNCH = os.path.join(REPO, 'scripts', 'pilot', 'launch_run.sh')
L1 = ('--prediction_type', 'direct', '--loss_type', 'l1')
L2 = ('--prediction_type', 'direct', '--loss_type', 'l2')


@pytest.fixture
def env(tmp_path):
    """Stub nvidia-smi (GPU free) and python (prints its args, so they land in run.log, then appends them to $STUB_ARGS)
    first on PATH; results under tmp_path/root; no RUN_NAME inherited from the caller."""
    stub = tmp_path / 'stub'
    stub.mkdir()
    (stub / 'nvidia-smi').write_text('#!/bin/bash\ncase "$*" in *memory.free*) echo 20000 ;; esac\n')
    (stub / 'python').write_text('#!/bin/bash\necho "$*"\necho "$*" >> "$STUB_ARGS"\n')
    for f in ('nvidia-smi', 'python'):
        os.chmod(stub / f, 0o755)
    e = {k: v for k, v in os.environ.items() if k != 'RUN_NAME'}
    e.update(PATH=f'{stub}:{os.environ["PATH"]}', RESIDUAL_WARMSTART_RESULTS=str(tmp_path / 'root'),
             PILOT_PY=str(stub / 'python'), STUB_ARGS=str(tmp_path / 'stub_args'))
    return e


def launch(env, run_name, mode, gpu, *extra):
    e = dict(env) if run_name is None else dict(env, RUN_NAME=run_name)
    return subprocess.run(['bash', LAUNCH, mode, str(gpu), *extra], env=e, capture_output=True, text=True, timeout=60)


def stub_lines(tmp_path, n):
    """The stub's argument lines once n complete ones are there (`echo >>` creates the file before it writes the line)."""
    path = tmp_path / 'stub_args'
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        text = path.read_text() if path.exists() else ''
        if text.endswith('\n') and text.count('\n') >= n:
            return text.splitlines()
        time.sleep(0.05)
    raise AssertionError(f'fewer than {n} complete stub lines in {path} after 10 s')


def wait_dead(run):
    """Until the process in run/pid has exited (launch_run.sh treats a pid that answers kill -0 as a live run)."""
    pid = int((run / 'pid').read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.05)
    raise AssertionError(f'stub pid {pid} still alive after 10 s')


def test_run_name_selects_the_run_directory_and_extra_args_pass_through(env, tmp_path):
    r = launch(env, 'regress_l2_d4_b64', 'regress', 1, *L2)
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / 'root' / 'runs' / 'regress_l2_d4_b64'
    assert (run / 'launches').read_text().strip() == '1'
    assert not (tmp_path / 'root' / 'runs' / 'regress_d4_b64').exists()
    a = stub_lines(tmp_path, 1)[0]
    assert '--mode regress' in a and '--device cuda:1' in a and f'--save_dir {run} ' in a
    assert a.endswith('--prediction_type direct --loss_type l2')


def test_a_relaunch_resumes_from_the_run_names_own_checkpoints(env, tmp_path):
    run = tmp_path / 'root' / 'runs' / 'regress_l2_d4_b64'
    assert launch(env, 'regress_l2_d4_b64', 'regress', 1, *L2).returncode == 0
    assert '--resume' not in stub_lines(tmp_path, 1)[0]
    wait_dead(run)                                                  # the run died ...
    (run / 'checkpoint_epoch_10.pth').write_bytes(b'x')             # ... after its epoch-10 checkpoint
    r = launch(env, 'regress_l2_d4_b64', 'regress', 1, *L2)        # Task 7: the same command relaunches it
    assert r.returncode == 0, r.stdout + r.stderr
    assert stub_lines(tmp_path, 2)[1].endswith(f'--save_dir {run} --resume auto --prediction_type direct --loss_type l2')
    assert (run / 'launches').read_text().strip() == '2'


def test_done_is_per_run(env, tmp_path):
    runs = tmp_path / 'root' / 'runs'
    (runs / 'regress_l1_d4_b64').mkdir(parents=True)
    (runs / 'regress_l1_d4_b64' / 'DONE').write_text('100\n')
    r = launch(env, 'regress_l1_d4_b64', 'regress', 0, *L1)
    assert r.returncode != 0 and 'DONE exists' in r.stdout, r.stdout + r.stderr
    assert not (runs / 'regress_l1_d4_b64' / 'launches').exists()
    r = launch(env, 'regress_l2_d4_b64', 'regress', 0, *L2)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (runs / 'regress_l2_d4_b64' / 'launches').read_text().strip() == '1'
    assert f'--save_dir {runs / "regress_l2_d4_b64"} ' in stub_lines(tmp_path, 1)[0]


def test_two_run_names_of_one_mode_keep_their_own_pid_launches_and_run_log(env, tmp_path):
    runs = tmp_path / 'root' / 'runs'
    assert launch(env, 'regress_l1_d4_b64', 'regress', 0, *L1).returncode == 0
    assert launch(env, 'regress_l2_d4_b64', 'regress', 1, *L2).returncode == 0
    stub_lines(tmp_path, 2)                                         # both stubs have written run.log, then stub_args
    assert sorted(os.listdir(runs)) == ['regress_l1_d4_b64', 'regress_l2_d4_b64']
    pids = set()
    for name, loss in (('regress_l1_d4_b64', 'l1'), ('regress_l2_d4_b64', 'l2')):
        run = runs / name
        assert (run / 'launches').read_text().strip() == '1'
        log = (run / 'run.log').read_text()
        assert f'--save_dir {run} ' in log and log.rstrip().endswith(f'--loss_type {loss}') and log.count('\n') == 1
        pids.add(int((run / 'pid').read_text()))
    assert len(pids) == 2


def test_an_empty_run_name_falls_back_to_mode_d4_b64(env, tmp_path):
    runs = tmp_path / 'root' / 'runs'
    r = launch(env, '', 'regress', 1, *L2)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (runs / 'regress_d4_b64' / 'launches').read_text().strip() == '1'
    assert not (runs / 'launches').exists()
    assert f'--save_dir {runs / "regress_d4_b64"} ' in stub_lines(tmp_path, 1)[0]


@pytest.mark.parametrize('run_name, mode, extra', [
    ('../regress_l2_d4_b64', 'regress', L2),            # leaves runs/ (and pins the ^ anchor)
    ('regress_l2/../../x', 'regress', L2),              # '/' (and pins the $ anchor)
    ('regress-l2-d4-b64', 'regress', L2),               # no <mode>_ prefix
    ('regress_', 'regress', L2),                        # nothing after the prefix
    ('regress_l2_d4_b64', 'standard', ()),              # an exported RUN_NAME reaching another mode's launch
])
def test_a_malformed_run_name_is_refused_before_anything_starts(env, tmp_path, run_name, mode, extra):
    r = launch(env, run_name, mode, 0, *extra)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and f"RUN_NAME='{run_name}' is not {mode}_" in out and 'not launching' in out, out
    assert not (tmp_path / 'root').exists() and not (tmp_path / 'stub_args').exists()


@pytest.mark.parametrize('run_name, extra, why', [
    ('regress_l2_d4_b64', ('--prediction_type', 'direct'), 'needs --loss_type'),        # would silently train l1
    ('', ('--prediction_type', 'direct'), 'needs --loss_type'),
    ('regress_l2_d4_b64', ('--prediction_type', 'direct', '--loss_type'), 'needs --loss_type'),
    ('regress_l2_d4_b64', L1, 'names l2 but --loss_type is l1'),
    ('regress_l1_d4_b64', L2, 'names l1 but --loss_type is l2'),
    ('regress_l2_s43_d4_b64', L1 + ('--seed', '43'), 'names l2 but --loss_type is l1'),
    ('regress_l2_d4_b64', L2 + ('--loss_type', 'l1'), 'names l2 but --loss_type is l1'),   # the last one wins
    ('regress_l2_d4_b64', ('--prediction_type', 'direct', '--loss_type=l1'), 'names l2 but --loss_type is l1'),
])
def test_a_regress_launch_without_a_matching_loss_type_is_refused(env, tmp_path, run_name, extra, why):
    r = launch(env, run_name, 'regress', 0, *extra)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and why in out and 'not launching' in out, out
    assert not (tmp_path / 'root').exists() and not (tmp_path / 'stub_args').exists()


@pytest.mark.parametrize('run_name, gpu, extra', [
    ('regress_l1_d4_b64', 0, L1),                                   # Task 7's planned launches
    ('regress_l2_s43_d4_b64', 1, L2 + ('--seed', '43')),
    ('regress_l2_d4_b64', 0, ('--prediction_type', 'direct', '--loss_type=l2')),
])
def test_the_planned_regress_launches_are_accepted(env, tmp_path, run_name, gpu, extra):
    r = launch(env, run_name, 'regress', gpu, *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / 'root' / 'runs' / run_name
    assert (run / 'launches').read_text().strip() == '1'
    a = stub_lines(tmp_path, 1)[0]
    assert '--mode regress' in a and f'--device cuda:{gpu}' in a and a.endswith(f'--save_dir {run} ' + ' '.join(extra))


# ----------------------------------------------------------------------------- final fix wave F2: seed <-> RUN_NAME
@pytest.mark.parametrize('run_name, mode, gpu, extra', [
    ('regress_l1_d4_b64', 'regress', 0, L1),                                  # Task 7 Step 1, verbatim
    ('regress_l2_d4_b64', 'regress', 1, L2),
    ('regress_l1_s43_d4_b64', 'regress', 0, L1 + ('--seed', '43')),           # Task 7 Step 5, verbatim
    ('regress_l2_s43_d4_b64', 'regress', 1, L2 + ('--seed', '43')),
    ('regress_l1_s43_d4_b64', 'regress', 0, L1 + ('--seed=43',)),
    ('regress_l1_s43_d4_b64', 'regress', 0, L1 + ('--seed', '42', '--seed', '43')),   # the last one wins
    ('regress_l1_d4_b64', 'regress', 0, L1 + ('--seed', '42')),               # the trainer's default, spelled out
    (None, 'residual', 0, ()),                                                # the Phase 1a pilot driver's launches
    (None, 'standard', 1, ()),
    (None, 'standard_xhat', 0, ()),
])
def test_the_planned_launches_pass_the_seed_guard(env, tmp_path, run_name, mode, gpu, extra):
    r = launch(env, run_name, mode, gpu, *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / 'root' / 'runs' / (run_name or f'{mode}_d4_b64')
    a = stub_lines(tmp_path, 1)[0]
    assert f'--mode {mode}' in a and f'--device cuda:{gpu}' in a
    assert a.endswith(f'--save_dir {run}' + ''.join(' ' + x for x in extra))


@pytest.mark.parametrize('run_name, extra, why', [
    ('regress_l1_s43_d4_b64', L1, "names seed 43 but --seed is ''"),          # would train seed 42 under the s43 name
    ('regress_l2_s43_d4_b64', L2 + ('--seed', '44'), "names seed 43 but --seed is '44'"),
    ('regress_l1_s43_d4_b64', L1 + ('--seed', '43', '--seed', '42'), "names seed 43 but --seed is '42'"),   # last wins
    ('regress_l1_s43_d4_b64', L1 + ('--seed=42',), "names seed 43 but --seed is '42'"),
    ('regress_l1_d4_b64_s43', L1, "names seed 43 but --seed is ''"),          # _s<N> at the end of the name
    ('regress_l1_d4_b64', L1 + ('--seed', '43'), "--seed 43 but RUN_NAME='regress_l1_d4_b64' names no _s<N>_"),
    ('', L1 + ('--seed=43',), "--seed 43 but RUN_NAME='' names no _s<N>_"),  # the fallback regress_d4_b64 is seed 42
    ('regress_l1_s43_d4_b64', L1 + ('--seed', '4x3'), "--seed '4x3' is not a non-negative integer"),
])
def test_a_regress_seed_that_contradicts_the_run_name_is_refused(env, tmp_path, run_name, extra, why):
    r = launch(env, run_name, 'regress', 0, *extra)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and why in out and 'not launching' in out, out
    assert not (tmp_path / 'root').exists() and not (tmp_path / 'stub_args').exists()


# ----------------------------------------------------------------------------- Phase 2a: half regressors and the new modes
D8 = ('--ds', '8')
XR = ('--xr_cache', '/tmp/xr_cache')


@pytest.mark.parametrize('run_name, mode, gpu, extra', [
    ('regress_l1_half1_d8_b64', 'regress', 0, L1 + D8 + ('--train_half', '1')),
    ('regress_l1_half2_d8_b64', 'regress', 1, L1 + D8 + ('--train_half=2',)),
    ('regress_l1_half1', 'regress', 0, L1 + ('--train_half', '1')),                   # _half<N> at the end
    ('regress_l1_sel_d8_b64', 'regress', 0, L1 + D8 + ('--train_half', '0')),         # 0 = all files, no token needed
    ('residual_reg_d8_b64', 'residual_reg', 0, D8 + XR),
    ('varhead_d8_b64', 'varhead', 1, ('--prediction_type', 'direct', '--loss_type', 'nll') + D8 + XR),
])
def test_the_phase_2a_launches_are_accepted(env, tmp_path, run_name, mode, gpu, extra):
    r = launch(env, run_name, mode, gpu, *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    a = stub_lines(tmp_path, 1)[0]
    assert f'--mode {mode}' in a and a.endswith(' '.join(extra))
    wait_dead(tmp_path / 'root' / 'runs' / run_name)


@pytest.mark.parametrize('run_name, mode, extra, why', [
    ('regress_l1_half1_d8_b64', 'regress', L1 + ('--train_half', '2'), 'names half 1'),
    ('regress_l1_half1_d8_b64', 'regress', L1, 'names half 1'),
    ('regress_l1_sel_d8_b64', 'regress', L1 + ('--train_half', '2'), 'names no _half'),
    ('varhead_d8_b64', 'varhead', ('--prediction_type', 'direct') + XR, 'needs --loss_type nll'),
    ('varhead_d8_b64', 'varhead', ('--prediction_type', 'direct', '--loss_type', 'l1') + XR, 'needs --loss_type nll'),
])
def test_phase_2a_contradictions_are_refused_before_anything_starts(env, tmp_path, run_name, mode, extra, why):
    r = launch(env, run_name, mode, 0, *extra)
    assert r.returncode == 1 and why in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / 'root').exists() and not (tmp_path / 'stub_args').exists()
