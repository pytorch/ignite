import pytest

from ignite.base.events import EventEnum
from ignite.base.events_driven import EventsDriven, EventsDrivenState, EventsDrivenWithState


class CustomEventsDrivenWithState(EventsDriven):
    def __init__(self) -> None:
        super(CustomEventsDrivenWithState, self).__init__()
        self._state = EventsDrivenState(engine=self)

    @property
    def state(self) -> EventsDrivenState:
        return self._state

    def register_events(self, *event_names, attr_to_events=None) -> None:
        super(CustomEventsDrivenWithState, self).register_events(*event_names)
        if attr_to_events is not None:
            for attribute, events in attr_to_events.items():
                self.state.update_attribute_mapping(attribute, events)


class ABCEvents(EventEnum):
    A_EVENT = "a_event"
    B_EVENT = "b_event"
    C_EVENT = "c_event"


def test_invalid_update_attribute_mapping():
    e = EventsDrivenState()

    with pytest.raises(TypeError, match=r"'attribute' must be a string, and `events` must be a list of Events."):
        e.update_attribute_mapping(attribute="a", events="b")

    with pytest.raises(TypeError, match=r"'attribute' must be a string, and `events` must be a list of Events."):
        e.update_attribute_mapping(attribute="a", events=5)

    with pytest.raises(TypeError, match=r"'attribute' must be a string, and `events` must be a list of Events."):
        e.update_attribute_mapping(attribute="a", events=(1, 2))


def test_invalid_event_names():
    e = EventsDriven()

    with pytest.raises(TypeError, match=r"Value at 0 of event_names should be a str or EventEnum"):
        e.register_events(5)

    with pytest.raises(TypeError, match=r"Value at 0 of event_names should be a str or EventEnum"):
        e.register_events(["a", "b"])


def test_not_allowed_events():
    e = EventsDriven()
    e.register_events("a", "b", "c", *ABCEvents)

    with pytest.raises(ValueError, match=r"Event f is not a valid event"):
        e._assert_allowed_event("f")

    with pytest.raises(ValueError, match=r"Event v is not a valid event"):
        e._assert_allowed_event("v")


def test_remove_not_existed_handler():
    e = EventsDriven()

    def handler():
        pass

    with pytest.raises(ValueError, match=r"Input event name 'dummy_event' does not exist"):
        e.remove_event_handler(handler, event_name="dummy_event")


def test_events_driven_basics():

    e = EventsDriven()
    assert len(e._allowed_events) == 0

    e.register_events("a", "b", "c", *ABCEvents)

    times_said_hello = [0]

    @e.on("a")
    def say_hello():
        times_said_hello[0] += 1

    e.fire_event("a")
    e.fire_event("a")
    assert times_said_hello[0] == 2

    times_handled_b_event = [0]

    def on_b_event():
        times_handled_b_event[0] += 1

    e.add_event_handler(ABCEvents.B_EVENT(every=2), on_b_event)

    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.B_EVENT)
    e.fire_event(ABCEvents.A_EVENT)
    e.fire_event(ABCEvents.A_EVENT)
    e.fire_event(ABCEvents.C_EVENT)
    assert times_handled_b_event[0] == 2
    assert e._allowed_events_counts[ABCEvents.A_EVENT] == 2
    assert e._allowed_events_counts[ABCEvents.B_EVENT] == 4
    assert e._allowed_events_counts[ABCEvents.C_EVENT] == 1


def test_events_driven_state_basics():
    state = EventsDrivenState()
    assert len(state._attr_to_events) == 0

    attr_to_events = {
        "a": [ABCEvents.A_EVENT],
        "b": [ABCEvents.B_EVENT],
        "c": [ABCEvents.C_EVENT],
    }
    for attribute, events in attr_to_events.items():
        state.update_attribute_mapping(attribute=attribute, events=events)

    assert len(state._attr_to_events) == 3
    assert state._attr_to_events["a"] == [
        ABCEvents.A_EVENT,
    ]
    assert state._attr_to_events["b"] == [
        ABCEvents.B_EVENT,
    ]
    assert state._attr_to_events["c"] == [
        ABCEvents.C_EVENT,
    ]

    state.update_attribute_mapping(attribute="epoch", events=["epoch_started", "epoch_completed"])
    assert state._attr_to_events["epoch"] == ["epoch_started", "epoch_completed"]


def test_events_driven_state_kwargs():
    state = EventsDrivenState(a=12, b=23)
    assert state.a == 12
    assert state.b == 23


def test_events_driven_state_derrived():
    class LikeEngineState(EventsDrivenState):
        attr_to_events = {
            "a": ["a_start", "a_end"],
            "b": ["b_start", "b_end"],
        }

        def __init__(self) -> None:
            super(LikeEngineState, self).__init__(attr_to_events=self.attr_to_events)

    s = LikeEngineState()

    assert s.attr_to_events == s._attr_to_events

    s.attr_to_events["c"] = ["c_start"]
    assert s.attr_to_events != s._attr_to_events

    s.update_attribute_mapping("c", ["c_start"])
    assert s.attr_to_events == s._attr_to_events


def test_basic_events_driven_with_state():

    attr_to_events = {
        "a": [ABCEvents.A_EVENT],
        "b": [ABCEvents.B_EVENT],
        "c": [ABCEvents.C_EVENT],
    }

    class TinyEngine(CustomEventsDrivenWithState):
        def __init__(self):
            super(TinyEngine, self).__init__()
            self.register_events(*ABCEvents, attr_to_events=attr_to_events)

        def _check(self):
            assert self.state.a == self._allowed_events_counts[ABCEvents.A_EVENT]
            assert self.state.b == self._allowed_events_counts[ABCEvents.B_EVENT]
            assert self.state.c == self._allowed_events_counts[ABCEvents.C_EVENT]

        def run(self, n, k, reset=True):
            if reset:
                self._reset_allowed_events_counts()
            self.fire_event(ABCEvents.A_EVENT)
            while self.state.b < n:
                self.fire_event(ABCEvents.B_EVENT)
                j = self.state.c % k
                while j < k:
                    j += 1
                    self.fire_event(ABCEvents.C_EVENT)
                    self._check()

    e = TinyEngine()

    for a, ev in attr_to_events.items():
        assert a in e.state._attr_to_events
        assert e.state._attr_to_events[a] == ev

    e.run(10, 20)
    assert e.state.a == 1
    assert e.state.b == 10
    assert e.state.c == 20 * 10

    e.state.a = 0
    e.state.b = 3
    e.state.c = 4

    assert e._allowed_events_counts[ABCEvents.A_EVENT] == 0
    assert e._allowed_events_counts[ABCEvents.B_EVENT] == 3
    assert e._allowed_events_counts[ABCEvents.C_EVENT] == 4

    e.run(10, 20, reset=False)
    assert e.state.a == 1
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 3)

    with pytest.raises(AttributeError, match=r"can't set attribute"):
        e.state = EventsDrivenState()

    e.state.a = 3
    e.state.b = 4
    e.state.c = 5

    assert e._allowed_events_counts[ABCEvents.A_EVENT] == 3
    assert e._allowed_events_counts[ABCEvents.B_EVENT] == 4
    assert e._allowed_events_counts[ABCEvents.C_EVENT] == 5

    e.run(10, 20, reset=False)
    assert e.state.a == 4
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 4)


def test_events_driven_with_state_mixed_events():
    class BCEvents(EventEnum):
        B_EVENT_STARTED = "b_event_started"
        B_EVENT_COMPLETED = "b_event_completed"
        C_EVENT_STARTED = "c_event_started"
        C_EVENT_COMPLETED = "c_event_completed"

    class AnotherTinyEngine(CustomEventsDrivenWithState):
        def __init__(self):
            super(AnotherTinyEngine, self).__init__()
            attr_to_events = {
                "b": [BCEvents.B_EVENT_STARTED, BCEvents.B_EVENT_COMPLETED],
                "c": [BCEvents.C_EVENT_STARTED, BCEvents.C_EVENT_COMPLETED],
            }
            self.register_events(*BCEvents, attr_to_events=attr_to_events)

        def _check(self):
            assert self.state.b == self._allowed_events_counts[BCEvents.B_EVENT_STARTED]
            assert self.state.c == self._allowed_events_counts[BCEvents.C_EVENT_STARTED]
            assert self.state.b - 1 == self._allowed_events_counts[BCEvents.B_EVENT_COMPLETED]
            assert self.state.c - 1 == self._allowed_events_counts[BCEvents.C_EVENT_COMPLETED]

        def run(self, n, k, reset=True):
            if reset:
                self._reset_allowed_events_counts()
            while self.state.b < n:
                self.fire_event(BCEvents.B_EVENT_STARTED)
                j = self.state.c % k
                while j < k:
                    j += 1
                    self.fire_event(BCEvents.C_EVENT_STARTED)
                    self._check()
                    self.fire_event(BCEvents.C_EVENT_COMPLETED)
                self.fire_event(BCEvents.B_EVENT_COMPLETED)

    e = AnotherTinyEngine()

    @e.on(BCEvents.B_EVENT_COMPLETED)
    def check_b():
        assert e.state.b == e._allowed_events_counts[BCEvents.B_EVENT_STARTED]

    @e.on(BCEvents.C_EVENT_COMPLETED)
    def check_c():
        assert e.state.c == e._allowed_events_counts[BCEvents.C_EVENT_STARTED]

    e.run(10, 20)
    assert e.state.b == 10
    assert e.state.c == 20 * 10

    e.state.b = 3
    e.state.c = 4

    assert e._allowed_events_counts[BCEvents.B_EVENT_STARTED] == 3
    assert e._allowed_events_counts[BCEvents.C_EVENT_STARTED] == 4
    assert e._allowed_events_counts[BCEvents.B_EVENT_COMPLETED] == 3
    assert e._allowed_events_counts[BCEvents.C_EVENT_COMPLETED] == 4

    e.run(10, 20, reset=False)
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 3)

    with pytest.raises(AttributeError, match=r"can't set attribute"):
        e.state = EventsDrivenState()

    e.state.b = 4
    e.state.c = 5

    assert e._allowed_events_counts[BCEvents.B_EVENT_STARTED] == 4
    assert e._allowed_events_counts[BCEvents.C_EVENT_STARTED] == 5
    assert e._allowed_events_counts[BCEvents.B_EVENT_STARTED] == 4
    assert e._allowed_events_counts[BCEvents.C_EVENT_STARTED] == 5

    e.run(10, 20, reset=False)
    assert e.state.b == 10
    assert e.state.c == 20 * (10 - 4)


def build_custom_engine_and_state():
    class TestEvents(EventEnum):
        TEST_EVENT_STARTED = "test_event_started"
        TEST_EVENT_COMPLETED = "test_event_completed"

    class CustomState(EventsDrivenState):

        attr_to_events = {"test_event": [TestEvents.TEST_EVENT_STARTED, TestEvents.TEST_EVENT_COMPLETED]}

        def __init__(self, engine=None, **kwargs) -> None:
            super(CustomState, self).__init__(engine=engine, attr_to_events=CustomState.attr_to_events, **kwargs)
            self.test_attr = 0

    class CustomEngine(EventsDrivenWithState):
        def __init__(self) -> None:
            super(CustomEngine, self).__init__()
            self._state = CustomState(engine=self)
            self.register_events(*TestEvents, attr_to_events=CustomState.attr_to_events)

        def register_events(self, *event_names, attr_to_events=None) -> None:
            super(CustomEngine, self).register_events(*event_names)
            if attr_to_events is not None:
                for attribute, events in attr_to_events.items():
                    self._state.update_attribute_mapping(attribute, events)

    toy_engine = CustomEngine()
    toy_engine.state.beta = 99
    toy_engine.state.alpha = 888

    assert (
        toy_engine._allowed_events_counts[TestEvents.TEST_EVENT_STARTED]
        == toy_engine._allowed_events_counts[TestEvents.TEST_EVENT_COMPLETED]
        == 888
    )
