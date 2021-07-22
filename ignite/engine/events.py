from typing import Any

from ignite.base.events import CallableEventWithFilter, EventEnum, EventsList, RemovableEventHandle

__all__ = ["CallableEventWithFilter", "EventEnum", "Events", "State", "EventsList", "RemovableEventHandle"]


class Events(EventEnum):
    """Events that are fired by the :class:`~ignite.engine.engine.Engine` during execution. Built-in events:

    - STARTED : triggered when engine's run is started
    - EPOCH_STARTED : triggered when the epoch is started
    - GET_BATCH_STARTED : triggered before next batch is fetched
    - GET_BATCH_COMPLETED : triggered after the batch is fetched
    - ITERATION_STARTED : triggered when an iteration is started
    - ITERATION_COMPLETED : triggered when the iteration is ended

    - DATALOADER_STOP_ITERATION : engine's specific event triggered when dataloader has no more data to provide

    - EXCEPTION_RAISED : triggered when an exception is encountered
    - TERMINATE_SINGLE_EPOCH : triggered when the run is about to end the current epoch,
      after receiving a :meth:`~ignite.engine.engine.Engine.terminate_epoch()` or
      :meth:`~ignite.engine.engine.Engine.terminate()` call.

    - TERMINATE : triggered when the run is about to end completely,
      after receiving :meth:`~ignite.engine.engine.Engine.terminate()` call.

    - EPOCH_COMPLETED : triggered when the epoch is ended. Note that this is triggered even
      when :meth:`~ignite.engine.engine.Engine.terminate_epoch()` is called.
    - COMPLETED : triggered when engine's run is completed

    The table below illustrates which events are triggered when various termination methods are called.

    .. list-table::
       :widths: 24 25 33 18
       :header-rows: 1

       * - Method
         - EVENT_COMPLETED
         - TERMINATE_SINGLE_EPOCH
         - TERMINATE
       * - no termination
         - ✔
         - ✗
         - ✗
       * - :meth:`~ignite.engine.engine.Engine.terminate_epoch()`
         - ✔
         - ✔
         - ✗
       * - :meth:`~ignite.engine.engine.Engine.terminate()`
         - ✗
         - ✔
         - ✔

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

    Since v0.4.0, user can also combine events with `|`-operator:

    .. code-block:: python

        events = Events.STARTED | Events.COMPLETED | Events.ITERATION_STARTED(every=3)
        engine = ...

        @engine.on(events)
        def call_on_events(engine):
            # do something

    Since v0.4.0, custom events defined by user should inherit from :class:`~ignite.engine.events.EventEnum` :

    .. code-block:: python

        class CustomEvents(EventEnum):
            FOO_EVENT = "foo_event"
            BAR_EVENT = "bar_event"
    """

    EPOCH_STARTED = "epoch_started"
    """triggered when the epoch is started."""
    EPOCH_COMPLETED = "epoch_completed"
    """Event attribute indicating epoch is ended."""

    STARTED = "started"
    """triggered when engine’s run is started."""
    COMPLETED = "completed"
    """"triggered when engine’s run is completed"""

    ITERATION_STARTED = "iteration_started"
    """triggered when an iteration is started."""
    ITERATION_COMPLETED = "iteration_completed"
    """triggered when the iteration is ended."""
    EXCEPTION_RAISED = "exception_raised"
    """triggered when an exception is encountered."""

    GET_BATCH_STARTED = "get_batch_started"
    """triggered before next batch is fetched."""
    GET_BATCH_COMPLETED = "get_batch_completed"
    """triggered after the batch is fetched."""

    DATALOADER_STOP_ITERATION = "dataloader_stop_iteration"
    """"engine’s specific event triggered when dataloader has no more data to provide"""
    TERMINATE = "terminate"
    """triggered when the run is about to end completely, after receiving terminate() call."""
    TERMINATE_SINGLE_EPOCH = "terminate_single_epoch"
    """triggered when the run is about to end the current epoch,
    after receiving a terminate_epoch() or terminate() call."""

    def __or__(self, other: Any) -> "EventsList":
        return EventsList() | self | other


# Import state here to keep BC compatibility
from ignite.engine.state import State
