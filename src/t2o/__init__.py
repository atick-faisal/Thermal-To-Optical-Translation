"""Detection-in-the-loop thermal→visible translation.

Package root is intentionally empty of behaviour: importing `t2o` must stay cheap and
side-effect free so that `t2o.cli` can configure logging before any submodule is loaded.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
