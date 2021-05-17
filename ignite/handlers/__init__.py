from typing import Any, Callable, Optional, Union

from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers.checkpoint import Checkpoint, DiskSaver, ModelCheckpoint
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.lr_finder import FastaiLRFinder
from ignite.handlers.param_scheduler import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    CyclicalScheduler,
    LinearCyclicalScheduler,
    LRScheduler,
    ParamGroupScheduler,
    ParamScheduler,
    PiecewiseLinear,
    create_lr_scheduler_with_warmup,
)
from ignite.handlers.stores import EpochOutputStore
from ignite.handlers.terminate_on_nan import TerminateOnNan
from ignite.handlers.time_limit import TimeLimit
from ignite.handlers.timing import Timer

__all__ = [
    "ModelCheckpoint",
    "Checkpoint",
    "DiskSaver",
    "Timer",
    "EarlyStopping",
    "TerminateOnNan",
    "global_step_from_engine",
    "TimeLimit",
    "EpochOutputStore",
    "ConcatScheduler",
    "CosineAnnealingScheduler",
    "LinearCyclicalScheduler",
    "LRScheduler",
    "ParamGroupScheduler",
    "ParamScheduler",
    "PiecewiseLinear",
    "CyclicalScheduler",
    "create_lr_scheduler_with_warmup",
    "FastaiLRFinder",
]


def global_step_from_engine(engine: Engine, custom_event_name: Optional[Events] = None) -> Callable:
    """Helper method to setup `global_step_transform` function using another engine.
    This can be helpful for logging trainer epoch/iteration while output handler is attached to an evaluator.

    Args:
        engine: engine which state is used to provide the global step
        custom_event_name: registered event name. Optional argument, event name to use.

    Returns:
        global step based on provided engine
    """

    def wrapper(_: Any, event_name: Events) -> int:
        if custom_event_name is not None:
            event_name = custom_event_name
        return engine.state.get_event_attrib_value(event_name)

    return wrapper
