import numbers
from typing import Any, Dict, Iterable, Optional, Union

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

    event_to_attr = {
        Events.GET_BATCH_STARTED: "iteration",
        Events.GET_BATCH_COMPLETED: "iteration",
        Events.ITERATION_STARTED: "iteration",
        Events.ITERATION_COMPLETED: "iteration",
        Events.EPOCH_STARTED: "epoch",
        Events.EPOCH_COMPLETED: "epoch",
        Events.STARTED: "epoch",
        Events.COMPLETED: "epoch",
    }  # type: Dict[Union[str, "Events", "CallableEventWithFilter"], str]

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

    def __getattr__(self, attr: str) -> Any:
        evnts = None
        if attr in self._attr_to_events:
            if attr == "epoch":
                evnts = self._attr_to_events["epoch"][0]  # Fetch EPOCH_STARTED
            elif attr == "iteration":
                evnts = self._attr_to_events["iteration"][2]  # Fetch ITERATION_STARTED
            else:
                evnts = self._attr_to_events[attr]

        if self.engine and evnts:
            if isinstance(evnts, list):
                # return max of available event counts
                counts = [self.engine._allowed_events_counts[e] for e in evnts]
                return max(counts)
            else:
                return self.engine._allowed_events_counts[evnts]

        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def _update_attrs(self) -> None:
        for value in self.event_to_attr.values():
            if not hasattr(self, value):
                setattr(self, value, 0)

    def get_event_attrib_value(self, event_name: Union[str, Events, CallableEventWithFilter]) -> int:
        """Get the value of Event attribute with given `event_name`."""
        if event_name not in State.event_to_attr:
            raise RuntimeError(f"Unknown event name '{event_name}'")
        return getattr(self, State.event_to_attr[event_name])

    def __repr__(self) -> str:
        s = "State:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str)):
                value = type(value)
            s += f"\t{attr}: {value}\n"
        return s
