""" ``trains_logger`` was renamed to ``clearml_logger``.

Note:
    ``trains_logger`` was renamed to ``clearml_logger``.
    Please refer to :mod:`~ignite.contrib.handlers.clearml_logger`.
"""
from ignite.contrib.handlers.clearml_logger import ClearMLLogger, ClearMLSaver

__all__ = [
    "TrainsLogger",
    "TrainsSaver",
]

TrainsLogger = ClearMLLogger
TrainsSaver = ClearMLSaver
