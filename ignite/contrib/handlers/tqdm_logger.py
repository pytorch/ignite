""" ``ignite.contrib.handlers.tqdm_logger`` was moved to ``ignite.handlers.tqdm_logger``.
Note:
    ``ignite.contrib.handlers.tqdm_logger`` was moved to ``ignite.handlers.tqdm_logger``.
    Please refer to :mod:`~ignite.handlers.tqdm_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/tqdm_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.tqdm_logger import ProgressBar

__all__ = [
    "ProgressBar",
]
ProgressBar = ProgressBar
