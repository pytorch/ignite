""" ``ignite.contrib.metrics.regression.manhattan_distance`` was moved to ``ignite.metrics.regression.manhattan_distance``. # noqa
Note:
    ``ignite.contrib.metrics.regression.manhattan_distance`` was moved to ``ignite.metrics.regression.manhattan_distance``. # noqa
    Please refer to :mod:`~ignite.metrics.regression.manhattan_distance`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/regression/manhattan_distance.py"
    f" and will be removed in version {removed_in}"
    if removed_in
    else "" ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.regression.manhattan_distance import ManhattanDistance

__all__ = ["ManhattanDistance"]

ManhattanDistance = ManhattanDistance
