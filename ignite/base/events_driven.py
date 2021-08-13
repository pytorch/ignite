import functools
import logging
import weakref
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from ignite.base.events import CallableEventWithFilter, EventEnum, EventsList, RemovableEventHandle
from ignite.engine.utils import _check_signature

if TYPE_CHECKING:
    from ignite.engine.events import Events


class EventsDriven:
    """Base class for events-driven engines without state. This class is mainly
    responsible for registering events and triggering events, it also keeps track of
    the allowed events, and how many times they have been triggered.

    Attributes:
        last_event_name: last event name triggered.

    Example:

        Register and fire custom events

        .. code-block:: python

            class ABEvents(EventEnum):
                A_EVENT = "a_event"
                B_EVENT = "b_event"

            e = EventsDriven()
            e.register_events("a", "b", *ABEvents)

            times_a_fired = [0]

            @e.on("a")
            def handle_a():
                times_a_fired[0] += 1

            times_b_fired = [0]

            def handle_b():
                times_b_fired[0] += 1

            e.add_event_handler(ABCEvents.B_EVENT(every=2), handle_b)

            # fire event a
            e.fire_event("a")
            e.fire_event("a")

            # fire event b
            e.fire_event(ABCEvents.B_EVENT)

    """

    def __init__(self) -> None:
        self._event_handlers = defaultdict(list)  # type: Dict[Any, List]

        self._allowed_events = []  # type: List[EventEnum]
        self._allowed_events_counts = {}  # type: Dict[Union[str, EventEnum], int]

        self.last_event_name = None  # type: Optional[Events]
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    def register_events(self, *event_names: Union[List[str], List[EventEnum]]) -> None:
        """Add events that can be fired.

        Args:
            event_names: Defines the name of the event being supported. New events can be a str
                or an object derived from :class:`~ignite.base.events.EventEnum`. See example below.
        """
        for index, e in enumerate(event_names):
            if not isinstance(e, (str, EventEnum)):
                raise TypeError(f"Value at {index} of event_names should be a str or EventEnum, but given {e}")
            self._allowed_events.append(e)
            self._allowed_events_counts[e] = 0

    def _handler_wrapper(self, handler: Callable, event_name: Any, event_filter: Callable) -> Callable:
        # signature of the following wrapper will be inspected during registering to check if engine is necessary
        # we have to build a wrapper with relevant signature : solution is functools.wraps
        @functools.wraps(handler)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            event = self._allowed_events_counts[event_name]
            if event_filter(self, event):
                return handler(*args, **kwargs)

        # setup input handler as parent to make has_event_handler work
        setattr(wrapper, "_parent", weakref.ref(handler))
        return wrapper

    def _assert_allowed_event(self, event_name: Any) -> None:
        if event_name not in self._allowed_events:
            self.logger.error(f"attempt to add event handler to an invalid event {event_name}")
            raise ValueError(f"Event {event_name} is not a valid event for this {self.__class__.__name__}.")

    def add_event_handler(self, event_name: Any, handler: Callable, *args: Any, **kwargs: Any) -> RemovableEventHandle:
        """Add an event handler to be executed when the specified event is fired.

        Args:
            event_name: An event or a list of events to attach the handler. Valid events are
                from :class:`~ignite.engine.events.Events` or any ``event_name`` added by
                :meth:`~ignite.base.events_driven.EventsDriven.register_events`.
            handler: the callable event handler that should be invoked. No restrictions on its signature.
                The first argument can be optionally `engine`, the :class:`~ignite.base.events_driven.EventsDriven`
                object, handler is bound to.
            *args: optional args to be passed to ``handler``.
            **kwargs: optional keyword args to be passed to ``handler``.

        Returns:
            :class:`~ignite.base.events.RemovableEventHandle`, which can be used to remove the handler.
        """
        if isinstance(event_name, EventsList):
            for e in event_name:
                self.add_event_handler(e, handler, *args, **kwargs)
            return RemovableEventHandle(event_name, handler, self)
        if (
            isinstance(event_name, CallableEventWithFilter)
            and event_name.filter != CallableEventWithFilter.default_event_filter
        ):
            event_filter = event_name.filter
            handler = self._handler_wrapper(handler, event_name, event_filter)

        self._assert_allowed_event(event_name)

        # TODO: remove this import after refactoring Events
        from ignite.engine.events import Events

        event_args = (Exception(),) if event_name == Events.EXCEPTION_RAISED else ()
        try:
            _check_signature(handler, "handler", self, *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, (self,) + args, kwargs))
        except ValueError:
            _check_signature(handler, "handler", *(event_args + args), **kwargs)
            self._event_handlers[event_name].append((handler, args, kwargs))
        self.logger.debug(f"added handler for event {event_name}")

        return RemovableEventHandle(event_name, handler, self)

    @staticmethod
    def _assert_non_filtered_event(event_name: Any) -> None:
        if (
            isinstance(event_name, CallableEventWithFilter)
            and event_name.filter != CallableEventWithFilter.default_event_filter
        ):
            raise TypeError(
                "Argument event_name should not be a filtered event, please use event without any event filtering"
            )

    def has_event_handler(self, handler: Callable, event_name: Optional[Any] = None) -> bool:
        """Check if the specified event has the specified handler.

        Args:
            handler: the callable event handler.
            event_name: The event the handler attached to. Set this
                to ``None`` to search all events.
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                return False
            events = [event_name]  # type: Union[List[Any], Dict[Any, List]]
        else:
            events = self._event_handlers
        for e in events:
            for h, _, _ in self._event_handlers[e]:
                if self._compare_handlers(handler, h):
                    return True
        return False

    @staticmethod
    def _compare_handlers(user_handler: Callable, registered_handler: Callable) -> bool:
        if hasattr(registered_handler, "_parent"):
            registered_handler = registered_handler._parent()  # type: ignore[attr-defined]
        return registered_handler == user_handler

    def remove_event_handler(self, handler: Callable, event_name: Any) -> None:
        """Remove event handler `handler` from registered handlers of the EventsDriven instance

        Args:
            handler: the callable event handler that should be removed
            event_name: The event the handler attached to.

        """
        if event_name not in self._event_handlers:
            raise ValueError(f"Input event name '{event_name}' does not exist")

        new_event_handlers = [
            (h, args, kwargs)
            for h, args, kwargs in self._event_handlers[event_name]
            if not self._compare_handlers(handler, h)
        ]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError(f"Input handler '{handler}' is not found among registered event handlers")
        self._event_handlers[event_name] = new_event_handlers

    def on(self, event_name: Any, *args: Any, **kwargs: Any) -> Callable:
        """Decorator shortcut for add_event_handler.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.events.Events`
                or any ``event_name`` added by :meth:`~ignite.base.events_driven.EventsDriven.register_events`.
            *args: optional args to be passed to `handler`.
            **kwargs: optional keyword args to be passed to `handler`.
        """

        def decorator(f: Callable) -> Callable:
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f

        return decorator

    def _fire_event(self, event_name: Any, *event_args: Any, **event_kwargs: Any) -> None:
        """Execute all the handlers associated with given event.

        This method executes all handlers associated with the event
        `event_name`. Optional positional and keyword arguments can be used to
        pass arguments to **all** handlers added with this event. These
        arguments updates arguments passed using :meth:`~ignite.base.events_driven.EventsDriven.add_event_handler`.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.base.events_driven.EventsDriven.register_events`.
            *event_args: optional args to be passed to all handlers.
            **event_kwargs: optional keyword args to be passed to all handlers.
        """
        self.logger.debug(f"firing handlers for event {event_name}")
        self.last_event_name = event_name
        self._allowed_events_counts[event_name] += 1
        for func, args, kwargs in self._event_handlers[event_name]:
            kwargs.update(event_kwargs)
            first, others = ((args[0],), args[1:]) if (args and args[0] == self) else ((), args)
            func(*first, *(event_args + others), **kwargs)

    def fire_event(self, event_name: Any) -> None:
        """Execute all the handlers associated with given event.

        Args:
            event_name: event for which the handlers should be executed. Valid
                events are from :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.base.events_driven.EventsDriven.register_events`.

        """
        self._assert_allowed_event(event_name)
        return self._fire_event(event_name)

    def _reset_allowed_events_counts(self) -> None:
        for k in self._allowed_events_counts:
            self._allowed_events_counts[k] = 0


class EventsDrivenState:
    """State for EventsDriven class. State attributes are automatically synchronized with
    EventsDriven counters.

    Args:
        engine: ignite engine :class:`~ignite.base.events_driven.EventsDriven` that used to access
            the allowed events and their counters.
        attr_to_events: mapping consists of the attributes mapped to a list events from
            :class:`~ignite.engine.events.Events` or any other custom events added
            by :meth:`~ignite.base.events_driven.EventsDriven.register_events`.
            Getting attribute values is done based the on first element in the list of the events.
        **kwargs: optional keyword args.

    """

    def __init__(
        self,
        engine: Optional[EventsDriven] = None,
        attr_to_events: Optional[Dict[str, List["Events"]]] = None,
        **kwargs: Any,
    ):

        self._attr_to_events = attr_to_events if attr_to_events else defaultdict(list)  # type: Dict[str, List[Events]]
        self._engine = engine  # type: Optional[EventsDriven]

    @property
    def engine(self) -> Optional[EventsDriven]:
        return self._engine

    def __getattr__(self, attr: str) -> Any:
        evnts = None
        if attr in self._attr_to_events:
            evnts = self._attr_to_events[attr]

        if self._engine and evnts:
            # return first of available event counts
            return self._engine._allowed_events_counts[evnts[0]]

        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __setattr__(self, attr: str, value: Any) -> None:
        if all([a in self.__dict__ for a in ["_engine", "_attr_to_events"]]) and self.__dict__["_engine"]:
            self__attr_to_events = self.__dict__["_attr_to_events"]
            evnts = None
            if attr in self__attr_to_events:
                evnts = self__attr_to_events[attr]
            self__engine = self.__dict__["_engine"]
            if self__engine and evnts:
                # Set all counters to provided value
                for e in evnts:
                    if e in self__engine._allowed_events:
                        self__engine._allowed_events_counts[e] = value
                return

        super().__setattr__(attr, value)

    def update_attribute_mapping(self, attribute: str, events: List["Events"]) -> None:
        """Maps a given attribute to a given events list. If the attribute already exists,
        then the corresponding events will be update to the new given events. If the attribute doesn't
        exist, it will be added. Also the list of events will be added in the same order, as the first
        event in the list will be used to get the attribute value.
        Args:
            attribute: a state attribute to add, or update if already existed. This attribute will be mapped
                to the given events. Also if the value of the attribute has been updated, all the counters
                of the mapped events will be updated accordingly.
            events: list of events that will be mapped to the attribute, these events are from
                :class:`~ignite.engine.events.Events` or any other custom events added by
                :meth:`~ignite.base.events_driven.EventsDriven.register_events`.
                The first event in this list will be used to get the attribute value.
        """
        if not isinstance(attribute, str) or not isinstance(events, list):
            raise TypeError("'attribute' must be a string, and `events` must be a list of Events.")
        self._attr_to_events.update({attribute: events})


class EventsDrivenWithState(EventsDriven):
    """Base class for events-driven engines with state as a property. This class also can
    register and trigger events, and keeping track of the allowed events.

    Attributes:
        state: object that is used to pass internal and user-defined state between event handlers.

    Example:

        Build custom engine and a custom state

        .. code-block:: python

            from ignite.base.events import EventEnum
            from ignite.base.events_driven import EventsDrivenState, EventsDrivenWithState

            class AlphaEvents(EventEnum):
                EventAlpha_Started = "EventAlpha_Started"
                EventAlpha_Ended = "EventAlpha_Ended"

            class ToyState(EventsDrivenState):
                attr_to_events = {
                    "alpha": [AlphaEvents.EventAlpha_Started, AlphaEvents.EventAlpha_Ended]
                }

                def __init__(self, engine=None, **kwargs) -> None:
                    super(ToyState, self).__init__(engine=engine, attr_to_events=ToyState.attr_to_events, **kwargs)
                    self.beta = 0

            class ToyEngine(EventsDrivenWithState):
                def __init__(self) -> None:
                    super(ToyEngine, self).__init__()
                    self._state = ToyState(engine=self)
                    self.register_events(*alpha_events, attr_to_events=ToyState.attr_to_events)

                def register_events(self, *event_names, attr_to_events=None) -> None:
                    super(ToyEngine, self).register_events(*event_names)
                    if attr_to_events is not None:
                        for attribute, events in attr_to_events.items():
                            self._state.update_attribute_mapping(attribute, events)

            toy_engine = ToyEngine()
            toy_engine.state.beta = 80
            toy_engine.state.alpha = 60
            # ....
    """

    def __init__(self) -> None:
        super(EventsDrivenWithState, self).__init__()
        self._state = EventsDrivenState(engine=self)

    @property
    def state(self) -> EventsDrivenState:
        return self._state
