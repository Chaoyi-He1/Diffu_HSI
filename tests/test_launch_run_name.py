"""launch_run.sh honours RUN_NAME (Phase 1a-prime Task 5): the run directory is RUN_NAME and the extra arguments reach
the trainer last."""
import os
import subprocess
import time

from data_loader.warmstart_paths import REPO

LAUNCH = os.path.join(REPO, 'scripts', 'pilot', 'launch_run.sh')


def test_run_name_selects_the_run_directory_and_extra_args_pass_through(tmp_path):
    stub = tmp_path / 'stub'
    stub.mkdir()
    (stub / 'nvidia-smi').write_text('#!/bin/bash\ncase "$*" in *memory.free*) echo 20000 ;; esac\n')
    (stub / 'python').write_text('#!/bin/bash\necho "$*" >> "$STUB_ARGS"\n')
    for f in ('nvidia-smi', 'python'):
        os.chmod(stub / f, 0o755)
    args_file = tmp_path / 'stub_args'
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', RESIDUAL_WARMSTART_RESULTS=str(tmp_path / 'root'),
               PILOT_PY=str(stub / 'python'), STUB_ARGS=str(args_file), RUN_NAME='regress_l2_d4_b64')
    r = subprocess.run(['bash', LAUNCH, 'regress', '1', '--prediction_type', 'direct', '--loss_type', 'l2'], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    run = tmp_path / 'root' / 'runs' / 'regress_l2_d4_b64'
    assert (run / 'launches').read_text().strip() == '1'
    assert not (tmp_path / 'root' / 'runs' / 'regress_d4_b64').exists()
    deadline = time.monotonic() + 10
    while not args_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    a = args_file.read_text()
    assert '--mode regress' in a and '--device cuda:1' in a and f'--save_dir {run}' in a
    assert a.rstrip().endswith('--prediction_type direct --loss_type l2')
