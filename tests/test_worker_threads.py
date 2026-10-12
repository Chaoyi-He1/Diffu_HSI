"""DataLoader workers run with one BLAS thread (final-review fix item 5).

The x0_hat estimate runs numpy linear algebra inside every DataLoader worker. A forked worker inherits the parent's
BLAS thread count (OpenBLAS defaults to one thread per core), so 16 workers x 32 BLAS threads oversubscribed the
machine in the pilot. The shell exports in the pilot wrappers protect only those launches; the worker_init_fn
`limit_worker_threads` protects every launch. torch's own worker loop limits only torch's intra-op threads.
"""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_info, threadpool_limits

import main_2d_residual_fsdp as tr
from data_loader.hfd_residual import limit_worker_threads, residual_collate_fn

CPU = torch.device('cpu')


def blas_threads():
    """The largest BLAS thread count among the BLAS libraries loaded in this process."""
    return max(p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas')


class ProbeSet(torch.utils.data.Dataset):
    """Each item reports the BLAS thread count of the process that loads it, shaped like (x64, y_d, x0_hat) HWC arrays
    so residual_collate_fn accepts it. Module level, so it pickles under the spawn start method."""
    sigma_d = 0.05

    def __len__(self):
        return 8

    def __getitem__(self, i):
        a = np.full((1, 1, 1), float(blas_threads()), dtype=np.float32)
        return a, a, a


def reported(loader):
    return [v for x, _, _ in loader for v in x.flatten().tolist()]


@pytest.fixture
def parent_at_four_blas_threads():
    """The parent process runs with 4 BLAS threads, so a worker that inherits the parent's setting reports 4."""
    with threadpool_limits(4, user_api='blas'):
        if blas_threads() != 4:
            pytest.skip('this BLAS cannot run 4 threads, so inheritance cannot be told apart from the limit')
        yield


def test_training_loader_workers_use_one_blas_thread(parent_at_four_blas_threads):
    args = SimpleNamespace(num_workers=2, seed=0)
    loader = tr.make_loader(ProbeSet(), args, batch_size=2, shuffle=False, rank=0, world=1, device=CPU)
    seen = reported(loader)
    assert len(seen) == 8 and set(seen) == {1.0}, f'workers reported BLAS thread counts {sorted(set(seen))}'
    assert blas_threads() == 4                                      # the limit applies in the workers, not the parent


def test_shuffled_training_loader_workers_use_one_blas_thread(parent_at_four_blas_threads):
    """The shuffled (DistributedSampler) loader is the one training iterates."""
    args = SimpleNamespace(num_workers=2, seed=0)
    loader = tr.make_loader(ProbeSet(), args, batch_size=2, shuffle=True, rank=0, world=1, device=CPU)
    loader.sampler.set_epoch(0)
    seen = reported(loader)
    assert len(seen) == 8 and set(seen) == {1.0}, f'workers reported BLAS thread counts {sorted(set(seen))}'


def test_without_the_worker_init_fn_workers_inherit_the_parent_setting(parent_at_four_blas_threads):
    """The control: the same loader without worker_init_fn reports 4, so the tests above measure the limit."""
    loader = torch.utils.data.DataLoader(ProbeSet(), batch_size=2, num_workers=2, collate_fn=residual_collate_fn)
    assert set(reported(loader)) == {4.0}


def test_time_train_step_loader_workers_use_one_blas_thread(parent_at_four_blas_threads):
    from scripts.time_train_step import make_loader
    seen = reported(make_loader(ProbeSet(), batch_size=2, num_workers=2))
    assert len(seen) == 8 and set(seen) == {1.0}, f'workers reported BLAS thread counts {sorted(set(seen))}'


def test_limit_worker_threads_pickles_and_works_under_spawn():
    """A lambda would fail here: spawn and forkserver pickle the worker_init_fn."""
    assert pickle.loads(pickle.dumps(limit_worker_threads)) is limit_worker_threads
    loader = torch.utils.data.DataLoader(ProbeSet(), batch_size=4, num_workers=1, collate_fn=residual_collate_fn,
                                         worker_init_fn=limit_worker_threads, multiprocessing_context='spawn')
    assert set(reported(loader)) == {1.0}
