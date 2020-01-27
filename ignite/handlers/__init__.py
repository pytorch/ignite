from typing import Callable, Any, Union
from enum import Enum

from ignite.engine import Engine
from ignite.engine.events import EventWithFilter, CallableEvents

from ignite.handlers.checkpoint import ModelCheckpoint, Checkpoint, DiskSaver
from ignite.handlers.timing import Timer
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.terminate_on_nan import TerminateOnNan

__all__ = [
    'ModelCheckpoint',
    'Checkpoint',
    'DiskSaver',
    'Timer',
    'EarlyStopping',
    'TerminateOnNan',
    'global_step_from_engine'
]


def global_step_from_engine(engine: Engine) -> Callable:
    """Helper method to setup `global_step_transform` function using another engine.
    This can be helpful for logging trainer epoch/iteration while output handler is attached to an evaluator.

    Args:
        engine (Engine): engine which state is used to provide the global step

    Returns:
        global step
    """

    def wrapper(_: Any, event_name: Union[EventWithFilter, CallableEvents, Enum]):
        return engine.state.get_event_attrib_value(event_name)

    return wrapper
