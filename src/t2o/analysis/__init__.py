"""Post-hoc analysis across runs. Reads finished artifacts; never trains anything."""

from __future__ import annotations

from t2o.analysis.aggregate import (
    AggregateReport,
    AggregationError,
    Arm,
    ArmSummary,
    PairedResult,
    RunRecord,
    aggregate,
    bootstrap_ci,
    load_run,
    metric_value,
    pair_runs,
    sign_flip_p_value,
    tidy_rows,
    write_csv,
)

__all__ = [
    "AggregateReport",
    "AggregationError",
    "Arm",
    "ArmSummary",
    "PairedResult",
    "RunRecord",
    "aggregate",
    "bootstrap_ci",
    "load_run",
    "metric_value",
    "pair_runs",
    "sign_flip_p_value",
    "tidy_rows",
    "write_csv",
]
