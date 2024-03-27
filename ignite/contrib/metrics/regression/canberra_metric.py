""" ``ignite.contrib.metrics.regression.canberra_metric`` was moved to ``ignite.metrics.regression.canberra_metric``. # noqa
Note:
    ``ignite.contrib.metrics.regression.canberra_metric`` was moved to ``ignite.metrics.regression.canberra_metric``. # noqa
    Please refer to :mod:`~ignite.metrics.regression.canberra_metric`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/regression/canberra_metric.py"
    f" and will be removed in version {removed_in}"
    if removed_in
    else "" ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.regression.canberra_metric import CanberraMetric

__all__ = ["CanberraMetric"]

CanberraMetric = CanberraMetric
