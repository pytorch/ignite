""" ``ignite.contrib.handlers.wandb_logger`` was moved to ``ignite.handlers.wandb_logger``.
Note:
    ``ignite.contrib.handlers.wandb_logger`` was moved to ``ignite.handlers.wandb_logger``.
    Please refer to :mod:`~ignite.handlers.wandb_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/wandb_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.utils import global_step_from_engine  # noqa
from ignite.handlers.wandb_logger import OptimizerParamsHandler, OutputHandler, WandBLogger

__all__ = ["WandBLogger", "OutputHandler", "OptimizerParamsHandler"]
WandBLogger = WandBLogger
OutputHandler = OutputHandler
OptimizerParamsHandler = OptimizerParamsHandler
