# tests/test_fit_b_kernel.py — the calibration cubes of the B+ kernel never touch the test subset or its source images
import json

import numpy as np
import pytest
import torch

from data_loader.warmstart_paths import SUBSET
from scripts.paired_stats import source_image
from scripts.sensor import fit_b_kernel as F


def test_calibration_cubes_are_disjoint_from_the_test_subset_by_source_image():
    sub = json.load(open(SUBSET))
    cal = F.calibration_files()
    test = set(sub['ddpm1000_subset'])
    assert cal and set(cal) <= set(sub['files']) and not (set(cal) & test)
    assert not ({source_image(f) for f in cal} & {source_image(f) for f in test})
    assert len({source_image(f) for f in cal}) == len(cal)                    # one cube per source image
    assert cal == sorted(cal) and F.calibration_files(n=5) == cal[:5]


def test_calibration_refuses_an_empty_pool(tmp_path):
    p = tmp_path / 'sub.json'
    p.write_text(json.dumps(dict(files=['A/Train/P1/3_1.mat', 'A/Train/P1/3_2.mat'], ddpm1000_subset=['A/Train/P1/3_1.mat'])))
    with pytest.raises(ValueError, match='no calibration cube'):
        F.calibration_files(str(p))


def test_an_exported_tree_states_its_commit_in_GIT_HEAD(tmp_path):
    from data_loader.warmstart_paths import git_head
    assert git_head(str(tmp_path)) is None                                    # neither a checkout nor an export
    (tmp_path / 'GIT_HEAD').write_text('0123456789abcdef0123456789abcdef01234567\n')
    assert git_head(str(tmp_path)) == '0123456789abcdef0123456789abcdef01234567'
    (tmp_path / 'GIT_HEAD').write_text('\n')
    assert git_head(str(tmp_path)) is None


class _Half(torch.nn.Module):
    """x0-prediction stub: half of x_t, so the guided draws depend on the seed."""
    def forward(self, x_t, cond, t):
        return 0.5 * x_t


def test_fit_pools_truth_minus_the_ensemble_mean(small_R, synthetic_prior):
    from model.diffusion_trainer import DiffusionTrainer
    from test_sd_eval import StubSensorDS
    ds = StubSensorDS(small_R, synthetic_prior, n=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
    tr = DiffusionTrainer(device=torch.device('cpu'), n_timesteps=1000, loss_type='l1', prediction_type='x0', snr_gamma=None)
    args = F.get_parser().parse_args(['--uncond', 'u', '--K', '2', '--pigdm_steps', '2'])
    S, rmse = F.fit(_Half(), tr, ds, loader, args, torch.device('cpu'))
    assert S.shape == (64, 64) and np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > -1e-10 and len(rmse) == 3
    assert np.trace(S) > 0
