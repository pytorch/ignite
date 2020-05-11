from typing import Any, Callable, Union

from ignite.engine import Engine
from ignite.engine.events import CallableEventWithFilter, EventEnum
from ignite.handlers.checkpoint import Checkpoint, DiskSaver, ModelCheckpoint
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.terminate_on_nan import TerminateOnNan
from ignite.handlers.timing import Timer

__all__ = [
    "ModelCheckpoint",
    "Checkpoint",
    "DiskSaver",
    "Timer",
    "EarlyStopping",
    "TerminateOnNan",
    "global_step_from_engine",
]


def global_step_from_engine(engine: Engine) -> Callable:
    """Helper method to setup `global_step_transform` function using another engine.
    This can be helpful for logging trainer epoch/iteration while output handler is attached to an evaluator.

    Args:
        engine (Engine): engine which state is used to provide the global step

    Returns:
        global step
    """

    def wrapper(_: Any, event_name: Union[EventEnum, CallableEventWithFilter]):
        return engine.state.get_event_attrib_value(event_name)

    return wrapper
