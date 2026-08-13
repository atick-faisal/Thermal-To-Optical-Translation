"""Every RNG the training path touches, seeded from one place.

E3 (TASKS.md M1.2) compares paired runs whose only intended difference is
``train.seed``, and decides the project's central claim on the paired per-seed
differences. That comparison is only meaningful if "only the seed differs" is audited
rather than assumed. Before this module the audit failed in three places: ``numpy.random``
and Python's global ``random`` were never seeded (ultralytics consumes the former globally
during every detector stage), and the train ``DataLoader`` had no ``worker_init_fn``, so at
``workers > 0`` each worker process started from an unseeded ``random``/numpy state.

The stakes are measured, not hypothetical: M1.2's gate put the run-to-run noise floor at
0.059 mAP50, which is about the size of the effect E3 is trying to detect. Unseeded RNG is
variance that cannot be attributed to anything.

**Deliberately absent: ``torch.use_deterministic_algorithms(True)`` and the cuDNN
determinism flags.** cuDNN non-determinism is precisely the variance E3 exists to quantify
across seeds -- forcing it away would hide the measurement and cost throughput (TASKS.md
M1.2 step 2).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed torch (CPU and CUDA), numpy, and Python's ``random``.

    Called from ``translators.build_translator`` (so a backbone's initial weights are
    determined by ``train.seed``, which is what makes that field's participation in
    ``config_hash()`` mean anything) and once per stage from ``engine.loop.run_loop``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Redundant on the current torch -- `torch.manual_seed` forwards here itself
    # (`torch/random.py::_manual_seed_impl`, verified against 2.12.1). Repeated anyway so
    # the set of RNGs this function covers can be read off the function, not off torch's
    # source; a future torch that stops forwarding would otherwise silently drop CUDA.
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for a ``DataLoader``: PyTorch's own documented recipe.

    ``worker_id`` is unused -- the ``DataLoader`` passes it positionally, so it is part of
    the signature, but torch has already folded it into each worker's *torch* RNG seed. The
    per-worker seed is therefore read back out of ``torch.initial_seed()`` rather than
    derived again here, which keeps ``random``/numpy in step with torch inside the worker
    instead of inventing a second, unrelated stream. The ``% 2**32`` is numpy's range limit.

    **Must stay a module-level function.** The training machine is native Windows, which
    spawns rather than forks (PLAN.md §3), so ``worker_init_fn`` is pickled to reach the
    worker. A lambda or a closure over ``Trainer`` would fail there and nowhere else --
    exactly the silent-hang class of bug §3 warns about.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
