""" ``ignite.contrib.handlers.base_logger`` was moved to ``ignite.handlers.base_logger``.
Note:
    ``ignite.contrib.handlers.base_logger`` was moved to ``ignite.handlers.base_logger``.
    Please refer to :mod:`~ignite.handlers.base_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/base_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.base_logger import (
    BaseHandler,
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsHandler,
    BaseWeightsScalarHandler,
)

__all__ = [
    "BaseHandler",
    "BaseWeightsHandler",
    "BaseOptimizerParamsHandler",
    "BaseOutputHandler",
    "BaseWeightsScalarHandler",
    "BaseLogger",
]
BaseHandler = BaseHandler
BaseWeightsHandler = BaseWeightsHandler
BaseOptimizerParamsHandler = BaseOptimizerParamsHandler
BaseOutputHandler = BaseOutputHandler
BaseWeightsScalarHandler = BaseWeightsScalarHandler
BaseLogger = BaseLogger
