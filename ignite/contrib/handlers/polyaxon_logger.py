""" ``ignite.contrib.handlers.polyaxon_logger`` was moved to ``ignite.handlers.polyaxon_logger``.
Note:
    ``ignite.contrib.handlers.polyaxon_logger`` was moved to ``ignite.handlers.polyaxon_logger``.
    Please refer to :mod:`~ignite.handlers.polyaxon_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/polyaxon_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.polyaxon_logger import OptimizerParamsHandler, OutputHandler, PolyaxonLogger
from ignite.handlers.utils import global_step_from_engine  # noqa

__all__ = ["PolyaxonLogger", "OutputHandler", "OptimizerParamsHandler"]
PolyaxonLogger = PolyaxonLogger
OutputHandler = OutputHandler
OptimizerParamsHandler = OptimizerParamsHandler
