from enum import Enum

from mock import MagicMock

from ignite.engine import Engine, Events
from ignite.engine.engine import CallableEvents, EventWithFilter

import pytest


def test_custom_events():
    class CustomEvents(Enum):
        TEST_EVENT = "test_event"

    # Dummy engine
    engine = Engine(lambda engine, batch: 0)
    engine.register_events(*CustomEvents)

    # Handle is never called
    handle = MagicMock()
    engine.add_event_handler(CustomEvents.TEST_EVENT, handle)
    engine.run(range(1))
    assert not handle.called

    # Advanced engine
    def process_func(engine, batch):
        engine.fire_event(CustomEvents.TEST_EVENT)

    engine = Engine(process_func)
    engine.register_events(*CustomEvents)

    # Handle should be called
    handle = MagicMock()
    engine.add_event_handler(CustomEvents.TEST_EVENT, handle)
    engine.run(range(1))
    assert handle.called


def test_custom_events_with_event_to_attr():

    class CustomEvents(Enum):
        TEST_EVENT = "test_event"

    custom_event_to_attr = {CustomEvents.TEST_EVENT: 'test_event'}

    # Dummy engine
    engine = Engine(lambda engine, batch: 0)
    engine.register_events(*CustomEvents, event_to_attr=custom_event_to_attr)

    # Handle is never called
    handle = MagicMock()
    engine.add_event_handler(CustomEvents.TEST_EVENT, handle)
    engine.run(range(1))
    assert hasattr(engine.state, 'test_event')
    assert engine.state.test_event == 0

    # Advanced engine
    def process_func(engine, batch):
        engine.fire_event(CustomEvents.TEST_EVENT)

    engine = Engine(process_func)
    engine.register_events(*CustomEvents, event_to_attr=custom_event_to_attr)

    def handle(engine):
        engine.state.test_event += 1

    engine.add_event_handler(CustomEvents.TEST_EVENT, handle)
    engine.run(range(25))
    assert engine.state.test_event == 25

    custom_event_to_attr = 'a'
    engine = Engine(lambda engine, batch: 0)
    with pytest.raises(ValueError):
        engine.register_events(*CustomEvents, event_to_attr=custom_event_to_attr)


def test_callable_events_with_wrong_inputs():

    with pytest.raises(ValueError, match=r"Only one of the input arguments should be specified"):
        Events.ITERATION_STARTED()

    with pytest.raises(ValueError, match=r"Only one of the input arguments should be specified"):
        Events.ITERATION_STARTED(event_filter="123", every=12)

    with pytest.raises(TypeError, match=r"Argument event_filter should be a callable"):
        Events.ITERATION_STARTED(event_filter="123")

    with pytest.raises(ValueError, match=r"Argument every should be integer and greater than zero"):
        Events.ITERATION_STARTED(every=-1)

    with pytest.raises(ValueError, match=r"but will be called with"):
        Events.ITERATION_STARTED(event_filter=lambda x: x)


def test_callable_events():

    assert isinstance(Events.ITERATION_STARTED.value, str)

    def foo(engine, event):
        return True

    ret = Events.ITERATION_STARTED(event_filter=foo)
    assert isinstance(ret, EventWithFilter)
    assert ret.event == Events.ITERATION_STARTED
    assert ret.filter == foo
    assert isinstance(Events.ITERATION_STARTED.value, str)

    # assert ret in Events
    assert Events.ITERATION_STARTED.name in "{}".format(ret)
    # assert ret in State.event_to_attr

    ret = Events.ITERATION_STARTED(every=10)
    assert isinstance(ret, EventWithFilter)
    assert ret.event == Events.ITERATION_STARTED
    assert ret.filter is not None

    # assert ret in Events
    assert Events.ITERATION_STARTED.name in "{}".format(ret)
    # assert ret in State.event_to_attr

    ret = Events.ITERATION_STARTED(once=10)
    assert isinstance(ret, EventWithFilter)
    assert ret.event == Events.ITERATION_STARTED
    assert ret.filter is not None

    # assert ret in Events
    assert Events.ITERATION_STARTED.name in "{}".format(ret)
    # assert ret in State.event_to_attr

    def _attach(e1, e2):
        assert id(e1) != id(e2)

    _attach(Events.ITERATION_STARTED(every=10), Events.ITERATION_COMPLETED(every=10))


def test_callable_events_every_eq_one():
    e = Events.ITERATION_STARTED(every=1)
    assert not isinstance(e, EventWithFilter)
    assert isinstance(e, Events)


def test_every_event_filter_with_engine():

    def _test(event_name, event_attr, every, true_num_calls):

        engine = Engine(lambda e, b: b)

        counter = [0, ]
        counter_every = [0, ]
        num_calls = [0, ]

        @engine.on(event_name(every=every))
        def assert_every(engine):
            counter_every[0] += every
            assert getattr(engine.state, event_attr) % every == 0
            assert counter_every[0] == getattr(engine.state, event_attr)
            num_calls[0] += 1

        @engine.on(event_name)
        def assert_(engine):
            counter[0] += 1
            assert getattr(engine.state, event_attr) == counter[0]

        d = list(range(100))
        engine.run(d, max_epochs=5)

        assert num_calls[0] == true_num_calls

    _test(Events.ITERATION_STARTED, "iteration", 10, 100 * 5 // 10)
    _test(Events.ITERATION_COMPLETED, "iteration", 10, 100 * 5 // 10)
    _test(Events.EPOCH_STARTED, "epoch", 2, 5 // 2)
    _test(Events.EPOCH_COMPLETED, "epoch", 2, 5 // 2)


def test_once_event_filter_with_engine():

    def _test(event_name, event_attr):

        engine = Engine(lambda e, b: b)

        once = 2
        counter = [0, ]
        num_calls = [0, ]

        @engine.on(event_name(once=once))
        def assert_once(engine):
            assert getattr(engine.state, event_attr) == once
            num_calls[0] += 1

        @engine.on(event_name)
        def assert_(engine):
            counter[0] += 1
            assert getattr(engine.state, event_attr) == counter[0]

        d = list(range(100))
        engine.run(d, max_epochs=5)

        assert num_calls[0] == 1

    _test(Events.ITERATION_STARTED, "iteration")
    _test(Events.ITERATION_COMPLETED, "iteration")
    _test(Events.EPOCH_STARTED, "epoch")
    _test(Events.EPOCH_COMPLETED, "epoch")


def test_custom_event_filter_with_engine():

    special_events = [1, 2, 5, 7, 17, 20]

    def custom_event_filter(engine, event):
        if event in special_events:
            return True
        return False

    def _test(event_name, event_attr, true_num_calls):

        engine = Engine(lambda e, b: b)

        num_calls = [0, ]

        @engine.on(event_name(event_filter=custom_event_filter))
        def assert_on_special_event(engine):
            assert getattr(engine.state, event_attr) == special_events.pop(0)
            num_calls[0] += 1

        d = list(range(50))
        engine.run(d, max_epochs=25)

        assert num_calls[0] == true_num_calls

    _test(Events.ITERATION_STARTED, "iteration", len(special_events))
    _test(Events.ITERATION_COMPLETED, "iteration", len(special_events))
    _test(Events.EPOCH_STARTED, "epoch", len(special_events))
    _test(Events.EPOCH_COMPLETED, "epoch", len(special_events))


def test_callable_event_bad_behaviour():

    special_events = [1, 2, 5, 7, 17, 20]

    def custom_event_filter(engine, event):
        if event in special_events:
            return True
        return False

    # Check bad behaviour
    engine = Engine(lambda e, b: b)
    counter = [0, ]

    # Modify events
    Events.ITERATION_STARTED(event_filter=custom_event_filter)

    @engine.on(Events.ITERATION_STARTED)
    def assert_all_iters(engine):
        counter[0] += 1
        assert engine.state.iteration == counter[0]

    d = list(range(50))
    engine.run(d, max_epochs=25)

    assert counter[0] == engine.state.iteration


def test_custom_callable_events():

    class CustomEvents(Enum):
        TEST_EVENT = "test_event"

    with pytest.raises(TypeError, match=r"object is not callable"):
        CustomEvents.TEST_EVENT(every=10)

    class CustomEvents2(CallableEvents, Enum):
        TEST_EVENT = "test_event"

    CustomEvents2.TEST_EVENT(every=10)


def test_custom_callable_events_with_engine():

    class CustomEvents(CallableEvents, Enum):
        TEST_EVENT = "test_event"

    event_to_attr = {
        CustomEvents.TEST_EVENT: "test_event"
    }

    special_events = [1, 2, 5, 7, 17, 20]

    def custom_event_filter(engine, event):
        if event in special_events:
            return True
        return False

    def _test(event_name, event_attr, true_num_calls):

        def update_fn(engine, batch):
            engine.state.test_event = engine.state.iteration
            engine.fire_event(CustomEvents.TEST_EVENT)

        engine = Engine(update_fn)
        engine.register_events(*CustomEvents, event_to_attr=event_to_attr)

        num_calls = [0, ]

        @engine.on(event_name(event_filter=custom_event_filter))
        def assert_on_special_event(engine):
            assert getattr(engine.state, event_attr) == special_events.pop(0)
            num_calls[0] += 1

        d = list(range(50))
        engine.run(d, max_epochs=25)

        assert num_calls[0] == true_num_calls

    _test(CustomEvents.TEST_EVENT, "test_event", len(special_events))
