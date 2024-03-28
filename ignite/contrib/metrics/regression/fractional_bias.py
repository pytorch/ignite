""" ``ignite.contrib.metrics.regression.fractional_bias`` was moved to ``ignite.metrics.regression.fractional_bias``. # noqa
Note:
    ``ignite.contrib.metrics.regression.fractional_bias`` was moved to ``ignite.metrics.regression.fractional_bias``. # noqa
    Please refer to :mod:`~ignite.metrics.regression.fractional_bias`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/regression/fractional_bias.py"
    f" and will be removed in version {removed_in}"
    if removed_in
    else "" ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.regression.fractional_bias import FractionalBias

__all__ = ["FractionalBias"]

FractionalBias = FractionalBias
