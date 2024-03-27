""" ``ignite.contrib.metrics.regression.mean_normalized_bias`` was moved to ``ignite.metrics.regression.mean_normalized_bias``. # noqa
Note:
    ``ignite.contrib.metrics.regression.mean_normalized_bias`` was moved to ``ignite.metrics.regression.mean_normalized_bias``. # noqa
    Please refer to :mod:`~ignite.metrics.regression.mean_normalized_bias`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/regression/mean_normalized_bias.py"
    f" and will be removed in version {removed_in}"
    if removed_in
    else "" ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.regression.mean_normalized_bias import MeanNormalizedBias

__all__ = ["MeanNormalizedBias"]

MeanNormalizedBias = MeanNormalizedBias
