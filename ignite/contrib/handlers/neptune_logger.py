""" ``ignite.contrib.handlers.neptune_logger`` was moved to ``ignite.handlers.neptune_logger``.
Note:
    ``ignite.contrib.handlers.neptune_logger`` was moved to ``ignite.handlers.neptune_logger``.
    Please refer to :mod:`~ignite.handlers.neptune_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/neptune_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.neptune_logger import (
    _INTEGRATION_VERSION_KEY,
    GradsScalarHandler,
    NeptuneLogger,
    NeptuneSaver,
    OptimizerParamsHandler,
    OutputHandler,
    WeightsScalarHandler,
)
from ignite.handlers.utils import global_step_from_engine  # noqa

__all__ = [
    "NeptuneLogger",
    "NeptuneSaver",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "GradsScalarHandler",
]
NeptuneLogger = NeptuneLogger
NeptuneSaver = NeptuneSaver
OptimizerParamsHandler = OptimizerParamsHandler
OutputHandler = OutputHandler
WeightsScalarHandler = WeightsScalarHandler
GradsScalarHandler = GradsScalarHandler
_INTEGRATION_VERSION_KEY = _INTEGRATION_VERSION_KEY
