""" ``ignite.contrib.metrics.average_precision`` was moved to ``ignite.metrics.average_precision``.
Note:
    ``ignite.contrib.metrics.average_precision`` was moved to ``ignite.metrics.average_precision``.
    Please refer to :mod:`~ignite.metrics.average_precision`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/metrics/average_precision.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.average_precision import AveragePrecision

__all__ = [
    "AveragePrecision",
]

AveragePrecision = AveragePrecision
