from typing import Any, Callable, Optional

from ignite.engine import Engine, State
from ignite.engine.events import Events


def global_step_from_engine(
    engine: Engine,
    custom_event_name: Optional[Events] = None,
    fallback_attr: str = "epoch",
) -> Callable:
    """Helper method to setup `global_step_transform` function using another engine.
    This can be helpful for logging trainer epoch/iteration while output handler is attached to an evaluator.

    If the provided event is not registered in ``State.event_to_attr``,
    the function falls back to using ``engine.state.<fallback_attr>``.


    Args:
        engine: engine which state is used to provide the global step
        custom_event_name: registered event name. Optional argument, event name to use.
        fallback_attr: ``State`` attribute used when event is not registered. Default, "epoch".

    Returns:
        Callable returning global step value.

    .. versionchanged:: 0.5.4
        added ``fallback_attr`` argument as fallback ``State`` attribute.
    """

    def wrapper(_: Any, event_name: Events) -> int:
        if custom_event_name is not None:
            event_name = custom_event_name

        if event_name not in State.event_to_attr:
            return getattr(engine.state, fallback_attr)

        return engine.state.get_event_attrib_value(event_name)

    return wrapper
