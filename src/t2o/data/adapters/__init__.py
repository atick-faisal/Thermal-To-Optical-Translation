"""Per-dataset conversion from a public raw layout into the internal representation.

Only datasets that are genuinely paired (visible + infrared, same scene) belong here --
PLAN.md §9's ``{split}/{visible,infrared}/{images,labels}`` contract has no single-modality
mode. CPLID (RGB-only) and HIT-UAV (IR-only) don't fit it and are out of scope.
"""

from __future__ import annotations

from t2o.data.adapters.common import AdapterError
from t2o.data.adapters.flir import adapt_flir
from t2o.data.adapters.llvip import adapt_llvip
from t2o.data.adapters.m3fd import adapt_m3fd
from t2o.data.adapters.msrs import adapt_msrs

__all__ = ["AdapterError", "adapt_flir", "adapt_llvip", "adapt_m3fd", "adapt_msrs"]
