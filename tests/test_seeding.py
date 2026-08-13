"""M1.2 step 2: every RNG the training path touches is seeded from one place."""

from __future__ import annotations

import pickle
import random

import numpy as np
import torch

from t2o.seeding import seed_everything, seed_worker


def _draw() -> tuple[float, float, float]:
    return (torch.rand(()).item(), float(np.random.random()), random.random())


def test_the_same_seed_reproduces_torch_numpy_and_random() -> None:
    seed_everything(7)
    first = _draw()

    seed_everything(7)
    assert _draw() == first


def test_a_different_seed_changes_every_generator() -> None:
    seed_everything(7)
    first = _draw()

    seed_everything(8)
    second = _draw()

    # Zipped rather than compared as tuples: a single generator silently ignoring the seed
    # is exactly the failure this milestone exists to rule out, and would still leave the
    # tuples unequal.
    assert all(a != b for a, b in zip(first, second, strict=True))


def test_seed_worker_derives_from_the_torch_seed_torch_already_set() -> None:
    """In a worker, `torch.initial_seed()` is the per-worker seed torch derived from the
    loader's generator -- so `random`/numpy end up in step with torch rather than on a
    second, unrelated stream.
    """
    torch.manual_seed(123)
    seed_worker(0)
    first = (float(np.random.random()), random.random())

    torch.manual_seed(123)
    seed_worker(0)
    assert (float(np.random.random()), random.random()) == first

    torch.manual_seed(456)
    seed_worker(0)
    assert (float(np.random.random()), random.random()) != first


def test_seed_worker_is_picklable() -> None:
    """The training machine is native Windows, which spawns rather than forks (PLAN.md §3),
    so `worker_init_fn` is pickled to reach each worker. A lambda or a closure would fail
    there and pass everywhere else.
    """
    assert pickle.loads(pickle.dumps(seed_worker)) is seed_worker
