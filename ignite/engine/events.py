
from typing import Callable, Optional, Union, Any

from enum import Enum
import numbers
import weakref

from ignite.engine.utils import _check_signature


__all__ = [
    'Events',
    'State'
]


class EventWithFilter:

    def __init__(self, event: Any, filter: Callable):
        if not callable(filter):
            raise TypeError("Argument filter should be callable")
        self.event = event
        self.filter = filter

    def __str__(self) -> str:
        return "<%s event=%s, filter=%r>" % (self.__class__.__name__, self.event, self.filter)


class CallableEvents:
    """Base class for Events implementing call operator and storing event filter. This class should be inherited
    for any custom events with event filtering feature:

    .. code-block:: python

        from ignite.engine.engine import CallableEvents

        class CustomEvents(CallableEvents, Enum):
            TEST_EVENT = "test_event"

        engine = ...
        engine.register_events(*CustomEvents, event_to_attr={CustomEvents.TEST_EVENT: "test_event"})

        @engine.on(CustomEvents.TEST_EVENT(every=5))
        def call_on_test_event_every(engine):
            # do something

    """

    def __call__(self, event_filter: Optional[Callable] = None,
                 every: Optional[int] = None, once: Optional[int] = None):

        if not ((event_filter is not None) ^ (every is not None) ^ (once is not None)):
            raise ValueError("Only one of the input arguments should be specified")

        if (event_filter is not None) and not callable(event_filter):
            raise TypeError("Argument event_filter should be a callable")

        if (every is not None) and not (isinstance(every, numbers.Integral) and every > 0):
            raise ValueError("Argument every should be integer and greater than zero")

        if (once is not None) and not (isinstance(once, numbers.Integral) and once > 0):
            raise ValueError("Argument every should be integer and positive")

        if every is not None:
            if every == 1:
                # Just return the event itself
                return self
            event_filter = CallableEvents.every_event_filter(every)

        if once is not None:
            event_filter = CallableEvents.once_event_filter(once)

        # check signature:
        _check_signature("engine", event_filter, "event_filter", "event")

        return EventWithFilter(self, event_filter)

    @staticmethod
    def every_event_filter(every: int) -> Callable:
        def wrapper(engine, event: int) -> bool:
            if event % every == 0:
                return True
            return False

        return wrapper

    @staticmethod
    def once_event_filter(once: int) -> Callable:
        def wrapper(engine, event: int) -> bool:
            if event == once:
                return True
            return False

        return wrapper


class Events(CallableEvents, Enum):
    """Events that are fired by the :class:`~ignite.engine.Engine` during execution.

    Since v0.3.0, Events become more flexible and allow to pass an event filter to the Engine:

    .. code-block:: python

        engine = Engine()

        # a) custom event filter
        def custom_event_filter(engine, event):
            if event in [1, 2, 5, 10, 50, 100]:
                return True
            return False

        @engine.on(Events.ITERATION_STARTED(event_filter=custom_event_filter))
        def call_on_special_event(engine):
             # do something on 1, 2, 5, 10, 50, 100 iterations

        # b) "every" event filter
        @engine.on(Events.ITERATION_STARTED(every=10))
        def call_every(engine):
            # do something every 10th iteration

        # c) "once" event filter
        @engine.on(Events.ITERATION_STARTED(once=50))
        def call_once(engine):
            # do something on 50th iteration

    Event filter function `event_filter` accepts as input `engine` and `event` and should return True/False.
    Argument `event` is the value of iteration or epoch, depending on which type of Events the function is passed.

    """
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"

    GET_BATCH_STARTED = "get_batch_started"
    GET_BATCH_COMPLETED = "get_batch_completed"


class State:
    """An object that is used to pass internal and user-defined state between event handlers. By default, state
    contains the following attributes:

    .. code-block:: python

        state.iteration         # 1-based, the first iteration is 1
        state.epoch             # 1-based, the first epoch is 1
        state.seed              # seed to set at each epoch
        state.dataloader        # data passed to engine
        state.epoch_length      # optional length of an epoch
        state.max_epochs        # number of epochs to run
        state.batch             # batch passed to `process_function`
        state.output            # output of `process_function` after a single iteration
        state.metrics           # dictionary with defined metrics if any

    """

    event_to_attr = {
        Events.GET_BATCH_STARTED: "iteration",
        Events.GET_BATCH_COMPLETED: "iteration",
        Events.ITERATION_STARTED: "iteration",
        Events.ITERATION_COMPLETED: "iteration",
        Events.EPOCH_STARTED: "epoch",
        Events.EPOCH_COMPLETED: "epoch",
        Events.STARTED: "epoch",
        Events.COMPLETED: "epoch",
    }

    def __init__(self, **kwargs):
        self.iteration = 0
        self.epoch = 0
        self.epoch_length = None
        self.max_epochs = None
        self.output = None
        self.batch = None
        self.metrics = {}
        self.dataloader = None
        self.seed = None

        for k, v in kwargs.items():
            setattr(self, k, v)

        for value in self.event_to_attr.values():
            if not hasattr(self, value):
                setattr(self, value, 0)

    def get_event_attrib_value(self, event_name: Union[EventWithFilter, CallableEvents, Enum]) -> int:
        if isinstance(event_name, EventWithFilter):
            event_name = event_name.event
        if event_name not in State.event_to_attr:
            raise RuntimeError("Unknown event name '{}'".format(event_name))
        return getattr(self, State.event_to_attr[event_name])

    def __repr__(self) -> str:
        s = "State:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str)):
                value = type(value)
            s += "\t{}: {}\n".format(attr, value)
        return s


class RemovableEventHandle:
    """A weakref handle to remove a registered event.

    A handle that may be used to remove a registered event handler via the
    remove method, with-statement, or context manager protocol. Returned from
    :meth:`~ignite.engine.Engine.add_event_handler`.


    Args:
        event_name: Registered event name.
        handler: Registered event handler, stored as weakref.
        engine: Target engine, stored as weakref.

    Example usage:

    .. code-block:: python

        engine = Engine()

        def print_epoch(engine):
            print("Epoch: {}".format(engine.state.epoch))

        with engine.add_event_handler(Events.EPOCH_COMPLETED, print_epoch):
            # print_epoch handler registered for a single run
            engine.run(data)

        # print_epoch handler is now unregistered
    """

    def __init__(self, event_name: Union[EventWithFilter, CallableEvents, Enum], handler: Callable, engine):
        self.event_name = event_name
        self.handler = weakref.ref(handler)
        self.engine = weakref.ref(engine)

    def remove(self) -> None:
        """Remove handler from engine."""
        handler = self.handler()
        engine = self.engine()

        if handler is None or engine is None:
            return

        if engine.has_event_handler(handler, self.event_name):
            engine.remove_event_handler(handler, self.event_name)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.remove()
