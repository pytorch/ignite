""" ``ignite.contrib.handlers.mlflow_logger`` was moved to ``ignite.handlers.mlflow_logger``.
Note:
    ``ignite.contrib.handlers.mlflow_logger`` was moved to ``ignite.handlers.mlflow_logger``.
    Please refer to :mod:`~ignite.handlers.mlflow_logger`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/mlflow_logger.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.mlflow_logger import MLflowLogger, OptimizerParamsHandler, OutputHandler
from ignite.handlers.utils import global_step_from_engine  # noqa

__all__ = ["MLflowLogger", "OutputHandler", "OptimizerParamsHandler"]
MLflowLogger = MLflowLogger
OutputHandler = OutputHandler
OptimizerParamsHandler = OptimizerParamsHandler
