""" ``ignite.contrib.handlers.visdom_logger`` was moved to ``ignite.handlers.visdom_logger``.
Note:
    ``ignite.contrib.handlers.visdom_logger`` was moved to ``ignite.handlers.visdom_logger``.
    Please refer to :mod:`~ignite.handlers.visdom_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/visdom_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.utils import global_step_from_engine  # noqa
from ignite.handlers.visdom_logger import (
    _DummyExecutor,
    GradsScalarHandler,
    OptimizerParamsHandler,
    OutputHandler,
    VisdomLogger,
    WeightsScalarHandler,
)

__all__ = [
    "VisdomLogger",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "GradsScalarHandler",
]
VisdomLogger = VisdomLogger
OptimizerParamsHandler = OptimizerParamsHandler
OutputHandler = OutputHandler
WeightsScalarHandler = WeightsScalarHandler
GradsScalarHandler = GradsScalarHandler
_DummyExecutor = _DummyExecutor
