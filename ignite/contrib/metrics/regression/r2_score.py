""" ``ignite.contrib.metrics.regression.r2_score`` was moved to ``ignite.metrics.regression.r2_score``. # noqa
Note:
    ``ignite.contrib.metrics.regression.r2_score`` was moved to ``ignite.metrics.regression.r2_score``. # noqa
    Please refer to :mod:`~ignite.metrics.regression.r2_score`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/regression/r2_score.py"
    f" and will be removed in version {removed_in}"
    if removed_in
    else "" ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.regression.r2_score import R2Score

__all__ = ["R2Score"]

R2Score = R2Score
