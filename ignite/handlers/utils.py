from typing import Any, Callable, Optional

from ignite.engine import Engine
from ignite.engine.events import Events


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
