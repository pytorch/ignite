""" ``ignite.contrib.handlers.lr_finder`` was moved to ``ignite.handlers.lr_finder``.
Note:
    ``ignite.contrib.handlers.lr_finder`` was moved to ``ignite.handlers.lr_finder``.
    Please refer to :mod:`~ignite.handlers.lr_finder`.
"""
import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/lr_finder.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.lr_finder import FastaiLRFinder

__all__ = [
    "FastaiLRFinder",
]

FastaiLRFinder = FastaiLRFinder
