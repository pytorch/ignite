""" ``ignite.contrib.metrics.regression.median_absolute_percentage_error`` was moved to ``ignite.metrics.regression.median_absolute_percentage_error``. # noqa
Note:
    ``ignite.contrib.metrics.regression.median_absolute_percentage_error`` was moved to ``ignite.metrics.regression.median_absolute_percentage_error``. # noqa
    Please refer to :mod:`~ignite.metrics.regression.median_absolute_percentage_error`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/regression/median_absolute_percentage_error.py"
    f" and will be removed in version {removed_in}"
    if removed_in
    else "" ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.regression.median_absolute_percentage_error import MedianAbsolutePercentageError

__all__ = ["MedianAbsolutePercentageError"]

MedianAbsolutePercentageError = MedianAbsolutePercentageError
