"""The prior, the gate subset and the results root are defined once (final-review fix item 15), so a refit or a new d
writes exactly the file that training and evaluation read."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import main_2d_residual_fsdp as tr
from data_loader import warmstart_paths as wp
from scripts import eval_warmstart as ew

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'


def test_training_and_harness_read_the_same_prior_and_subset():
    assert tr.PRIOR == ew.PRIOR == wp.PRIOR
    assert tr.SUBSET == ew.SUBSET == wp.SUBSET
    assert tr.RESULTS_ROOT == ew.RESULTS_ROOT == wp.RESULTS_ROOT
    assert wp.PRIOR == os.path.join(wp.RESULTS_ROOT, 'linear_prior_R1.npz')
    assert wp.SUBSET == os.path.join(REPO, 'tests', 'data', 'val_subset_200.json') and os.path.exists(wp.SUBSET)
    assert tr.get_parser().parse_args(['--mode', 'residual', '--save_dir', 'x']).prior == wp.PRIOR


def test_phase0_scripts_write_and_read_the_prior_training_reads():
    from scripts import compute_residual_scale, fit_linear_prior, time_train_step
    assert fit_linear_prior.get_parser().parse_args([]).out == wp.PRIOR             # a refit overwrites what training reads
    assert compute_residual_scale.get_parser().parse_args([]).prior == wp.PRIOR      # a new d lands in that same file
    assert time_train_step.PRIOR == wp.PRIOR
    assert time_train_step.TIMING == os.path.join(wp.RESULTS_ROOT, 'timing.json')


def test_shared_helpers_are_defined_once():
    assert tr.file_sha256 is wp.file_sha256 and tr.git_head is wp.git_head


def test_gate_output_name_is_defined_once():
    """The gate writes, and the pilot driver looks for, the one name defined here (stdlib only, so the driver needs no numpy)."""
    from scripts import gate_1a
    assert gate_1a.default_out_name is wp.gate_out_name
    assert (gate_1a.DEFAULT_MAX_NFE, gate_1a.DEFAULT_K) == (wp.GATE_DEFAULT_MAX_NFE, wp.GATE_DEFAULT_K) == (10, 2.0)
    assert wp.gate_out_name('residual_d4_b64') == 'gate_1a__residual_d4_b64.json'
    assert wp.gate_out_name('residual_d4_b64', False, 20, 1.0, True) == 'gate_1a__residual_d4_b64__nodone__nfe20__k1__1seed.json'


def test_results_root_honours_the_environment(tmp_path):
    """Every consumer follows $RESIDUAL_WARMSTART_RESULTS (read at import time) to the same prior."""
    code = ('import json, main_2d_residual_fsdp as tr\n'
            'from data_loader import warmstart_paths as wp\n'
            'from scripts import compute_residual_scale as c, eval_warmstart as ew, fit_linear_prior as f, time_train_step as t\n'
            "print(json.dumps(dict(wp=wp.PRIOR, tr=tr.PRIOR, tr_cli=tr.get_parser().parse_args(['--mode', 'residual', '--save_dir', 'x']).prior,\n"
            '                      ew=ew.PRIOR, fit=f.get_parser().parse_args([]).out, scale=c.get_parser().parse_args([]).prior,\n'
            '                      time_step=t.PRIOR, eval_dir=wp.EVAL_DIR)))\n')
    env = dict(os.environ, RESIDUAL_WARMSTART_RESULTS=str(tmp_path), CUDA_VISIBLE_DEVICES='')
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    want = os.path.join(str(tmp_path), 'linear_prior_R1.npz')
    assert got.pop('eval_dir') == os.path.join(str(tmp_path), 'eval')
    assert got == {k: want for k in got}


def test_default_results_root_is_the_absolute_shared_one():
    env = {k: v for k, v in os.environ.items() if k != 'RESIDUAL_WARMSTART_RESULTS'}
    code = 'from data_loader import warmstart_paths as wp; print(wp.RESULTS_ROOT)'
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == DEFAULT_ROOT


def test_paths_module_has_no_heavy_imports():
    """The driver, the gate and shell helpers may import it; it must not pull in torch or numpy."""
    code = 'import sys; from data_loader import warmstart_paths; print(sorted(m for m in ("torch", "numpy", "scipy") if m in sys.modules))'
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]'


def test_every_key_check_meta_compares_is_written_by_checkpoint_meta():
    args = tr.get_parser().parse_args(['--mode', 'residual', '--save_dir', 'x'])
    meta = tr.checkpoint_meta(args, 0.05, 3e-7, 'x' * 64)
    assert set(ew.META_ARGS) | {'sigma_d', 'prior_sha256'} <= set(meta)
    assert set(tr.RESUME_META_KEYS) <= set(meta)
    eval_args = SimpleNamespace(ckpt='c.pth', **{k: getattr(args, k) for k in ew.META_ARGS})
    assert ew.check_meta({'meta': meta}, eval_args, 0.05, 'x' * 64) is True


def test_prior_file_is_the_phase0_prior():
    """The prior every consumer reads is the one the Phase 0 baselines were computed with (sha pinned in
    tests/data/baseline_phase0.json)."""
    if not os.path.exists(ew.PRIOR):
        pytest.skip(f'no prior at {ew.PRIOR}')
    rec = json.load(open(os.path.join(REPO, 'tests', 'data', 'baseline_phase0.json')))
    pinned = {rec[k]['prior_sha256'] for k in rec}
    assert len(pinned) == 1 and rec['d4']['prior_sha256'].startswith('1ad9108f')
    assert wp.file_sha256(ew.PRIOR) == rec['d4']['prior_sha256']
