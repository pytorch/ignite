""" ``ignite.contrib.metrics.regression.mean_error`` was moved to ``ignite.metrics.regression.mean_error``. # noqa
Note:
    ``ignite.contrib.metrics.regression.mean_error`` was moved to ``ignite.metrics.regression.mean_error``. # noqa
    Please refer to :mod:`~ignite.metrics.regression.mean_error`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/regression/mean_error.py"
    f" and will be removed in version {removed_in}"
    if removed_in
    else "" ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.regression.mean_error import MeanError

__all__ = ["MeanError"]

MeanError = MeanError
