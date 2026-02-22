from typing import Any, Callable, Optional

from ignite.engine import Engine
from ignite.engine.events import Events


def global_step_from_engine(engine: Engine, custom_event_name: Optional[Events] = None) -> Callable:
    """Helper method to setup `global_step_transform` function using another engine.
    This can be helpful for logging trainer epoch/iteration while output handler is attached to an evaluator.

    If the provided event is not registered in ``State.event_to_attr``,
    the function falls back to using ``engine.state.epoch``.
    """

    def wrapper(_: Any, event_name: Events) -> int:
        if custom_event_name is not None:
            event_name = custom_event_name

        try:
            return engine.state.get_event_attrib_value(event_name)
        except RuntimeError:
            return engine.state.epoch

    return wrapper
