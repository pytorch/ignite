""" ``ignite.contrib.metrics.cohen_kappa`` was moved to ``ignite.metrics.cohen_kappa``.
Note:
    ``ignite.contrib.metrics.cohen_kappa`` was moved to ``ignite.metrics.cohen_kappa``.
    Please refer to :mod:`~ignite.metrics.cohen_kappa`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/cohen_kappa.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.cohen_kappa import CohenKappa

__all__ = [
    "CohenKappa",
]

CohenKappa = CohenKappa
