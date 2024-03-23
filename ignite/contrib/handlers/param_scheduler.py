""" ``ignite.contrib.handlers.param_scheduler`` was moved to ``ignite.handlers.param_scheduler``.
Note:
    ``ignite.contrib.handlers.param_scheduler`` was moved to ``ignite.handlers.param_scheduler``.
    Please refer to :mod:`~ignite.handlers.param_scheduler`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to /ignite/handlers/param_scheduler.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.handlers.param_scheduler import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
    CyclicalScheduler,
    LinearCyclicalScheduler,
    LRScheduler,
    ParamGroupScheduler,
    ParamScheduler,
    PiecewiseLinear,
)

__all__ = [
    "ConcatScheduler",
    "CosineAnnealingScheduler",
    "LinearCyclicalScheduler",
    "LRScheduler",
    "ParamGroupScheduler",
    "ParamScheduler",
    "PiecewiseLinear",
    "CyclicalScheduler",
    "create_lr_scheduler_with_warmup",
]

ConcatScheduler = ConcatScheduler
CosineAnnealingScheduler = CosineAnnealingScheduler
LinearCyclicalScheduler = LinearCyclicalScheduler
LRScheduler = LRScheduler
ParamGroupScheduler = ParamGroupScheduler
ParamScheduler = ParamScheduler
PiecewiseLinear = PiecewiseLinear
CyclicalScheduler = CyclicalScheduler
create_lr_scheduler_with_warmup = create_lr_scheduler_with_warmup
