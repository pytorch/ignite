""" ``ignite.contrib.metrics.precision_recall_curve`` was moved to ``ignite.metrics.precision_recall_curve``.
Note:
    ``ignite.contrib.metrics.precision_recall_curve`` was moved to ``ignite.metrics.precision_recall_curve``.
    Please refer to :mod:`~ignite.metrics.precision_recall_curve`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/precision_recall_curve.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.precision_recall_curve import precision_recall_curve_compute_fn, PrecisionRecallCurve

__all__ = [
    "PrecisionRecallCurve",
    "precision_recall_curve_compute_fn",
]


PrecisionRecallCurve = PrecisionRecallCurve
precision_recall_curve_compute_fn = precision_recall_curve_compute_fn
