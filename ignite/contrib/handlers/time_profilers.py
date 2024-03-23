""" ``ignite.contrib.handlers.time_profilers.py`` was moved to ``ignite.handlers.time_profilers``.
Note:
    ``ignite.contrib.handlers.time_profilers`` was moved to ``ignite.handlers.time_profilers``.
    Please refer to :mod:`~ignite.handlers.time_profilers`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/time_profilers.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.time_profilers import BasicTimeProfiler, HandlersTimeProfiler

__all__ = [
    "BasicTimeProfiler",
    "HandlersTimeProfiler",
]

BasicTimeProfiler = BasicTimeProfiler
HandlersTimeProfiler = HandlersTimeProfiler
