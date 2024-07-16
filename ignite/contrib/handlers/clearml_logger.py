""" ``ignite.contrib.handlers.clearml_logger`` was moved to ``ignite.handlers.clearml_logger``.
Note:
    ``ignite.contrib.handlers.clearml_logger`` was moved to ``ignite.handlers.clearml_logger``.
    Please refer to :mod:`~ignite.handlers.clearml_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/clearml_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.clearml_logger import (
    ClearMLLogger,
    ClearMLSaver,
    GradsHistHandler,
    GradsScalarHandler,
    OptimizerParamsHandler,
    OutputHandler,
    WeightsHistHandler,
    WeightsScalarHandler,
)
from ignite.handlers.utils import global_step_from_engine  # noqa

__all__ = [
    "ClearMLLogger",
    "ClearMLSaver",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "WeightsHistHandler",
    "GradsScalarHandler",
    "GradsHistHandler",
]
ClearMLLogger = ClearMLLogger
ClearMLSaver = ClearMLSaver
OptimizerParamsHandler = OptimizerParamsHandler
OutputHandler = OutputHandler
WeightsScalarHandler = WeightsScalarHandler
WeightsHistHandler = WeightsHistHandler
GradsScalarHandler = GradsScalarHandler
GradsHistHandler = GradsHistHandler
