"""Shared locations and provenance helpers of the residual warm-start project (spec 2026-10-08), defined once.

The training entry point, the evaluation harness, the Phase 0 scripts and the tests import these, so a refit of the
prior or a new d writes exactly the file that training and evaluation read.

  RESULTS_ROOT  $RESIDUAL_WARMSTART_RESULTS, default /data/chaoyi_he/HSI/Diffu/results/residual_warmstart (absolute, so
                every worktree reads the same prior); read once, at import time
  PRIOR         RESULTS_ROOT/linear_prior_R1.npz (scripts/fit_linear_prior.py, scripts/compute_residual_scale.py)
  EVAL_DIR      RESULTS_ROOT/eval
  SUBSET        tests/data/val_subset_200.json of this checkout (the 200 gate cubes; tracked in git)
  DATA_ROOT     $HFD_DATA_ROOT, default the HFD100 .mat dataset
  gate_out_name the Phase 1a gate's output file name (scripts/gate_1a.py writes it as default_out_name; the pilot
                driver looks for it), with the gate's default thresholds GATE_DEFAULT_MAX_NFE and GATE_DEFAULT_K

Standard library only: importing this module must not pull in torch or numpy.
"""
import hashlib
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
RESULTS_ROOT = os.environ.get('RESIDUAL_WARMSTART_RESULTS', '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart')
PRIOR = os.path.join(RESULTS_ROOT, 'linear_prior_R1.npz')
EVAL_DIR = os.path.join(RESULTS_ROOT, 'eval')
SUBSET = os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')
GATE_DEFAULT_MAX_NFE, GATE_DEFAULT_K = 10, 2.0


def gate_out_name(run_name_b, require_done=True, max_nfe=GATE_DEFAULT_MAX_NFE, k=GATE_DEFAULT_K, allow_single_seed=False):
    """The gate file's name next to B's evaluation. A run that peeks (no DONE required), uses other thresholds or allows a
    single seed must not overwrite the official gate file."""
    return (f'gate_1a__{run_name_b}' + ('' if require_done else '__nodone') + ('' if max_nfe == GATE_DEFAULT_MAX_NFE else f'__nfe{max_nfe}')
            + ('' if k == GATE_DEFAULT_K else f'__k{k:g}') + ('__1seed' if allow_single_seed else '') + '.json')


def file_sha256(path):
    """sha256 of a file's bytes (provenance of the prior npz)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_head(repo=REPO):
    """The checkout's HEAD commit, or None when git is unavailable (best effort; uncommitted changes are not recorded)."""
    try:
        r = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None
