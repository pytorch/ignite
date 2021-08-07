import numbers
from typing import Any, Dict, Iterable, List, Optional, Union

from torch.utils.data import DataLoader

from ignite.base.events import CallableEventWithFilter
from ignite.base.events_driven import EventsDrivenState
from ignite.engine.events import Events

__all__ = [
    "State",
]


class State(EventsDrivenState):
    """An object that is used to pass internal and user-defined state between event handlers. By default, state
    contains the following attributes:

    .. code-block:: python

        state.iteration         # 1-based, the first iteration is 1
        state.epoch             # 1-based, the first epoch is 1
        state.seed              # seed to set at each epoch
        state.dataloader        # data passed to engine
        state.epoch_length      # optional length of an epoch
        state.max_epochs        # number of epochs to run
        state.max_iters         # number of iterations to run
        state.batch             # batch passed to `process_function`
        state.output            # output of `process_function` after a single iteration
        state.metrics           # dictionary with defined metrics if any
        state.times             # dictionary with total and per-epoch times fetched on
                                # keys: Events.EPOCH_COMPLETED.name and Events.COMPLETED.name

    Args:
        kwargs: keyword arguments to be defined as State attributes.
    """

    # list of events in attr_to_events must be in this same order,
    # as we want to get iteration by Events.ITERATION_STARTED
    # and epoch by Events.EPOCH_STARTED
    attr_to_events = {
        "iteration": [
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
        ],
        "epoch": [Events.EPOCH_STARTED, Events.EPOCH_COMPLETED, Events.STARTED, Events.COMPLETED],
    }  # type: Dict[str, List[Union["Events", "CallableEventWithFilter"]]]

    def __init__(self, **kwargs: Any) -> None:
        super(State, self).__init__(**kwargs)

        self.epoch_length = None  # type: Optional[int]
        self.max_epochs = None  # type: Optional[int]
        self.max_iters = None  # type: Optional[int]
        self.output = None  # type: Optional[int]
        self.batch = None  # type: Optional[int]
        self.metrics = {}  # type: Dict[str, Any]
        self.dataloader = None  # type: Optional[Union[DataLoader, Iterable[Any]]]
        self.seed = None  # type: Optional[int]
        self.times = {
            Events.EPOCH_COMPLETED.name: None,
            Events.COMPLETED.name: None,
        }  # type: Dict[str, Optional[float]]

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Update the attributes if it's a standalone state
        # to keep BC compatibility
        if self.engine is None:
            self._update_attrs()

    def _update_attrs(self) -> None:
        for key in self.attr_to_events.keys():
            if not hasattr(self, key):
                setattr(self, key, 0)

    def get_event_attrib_value(self, event_name: Union[str, Events, CallableEventWithFilter]) -> int:
        """Get the value of Event attribute with given `event_name`."""
        event_to_attr = {}
        for attribute, events in State.attr_to_events.items():
            for event in events:
                event_to_attr[event] = attribute

        if event_name not in event_to_attr:
            raise RuntimeError(f"Unknown event name '{event_name}'")
        return getattr(self, event_to_attr[event_name])  # type: ignore

    def __repr__(self) -> str:
        s = "State:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str)):
                value = type(value)
            s += f"\t{attr}: {value}\n"
        return s
