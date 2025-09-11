from enum import Enum
from unittest.mock import MagicMock

import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.engine.events import CallableEventWithFilter, EventEnum, EventsList


def test_custom_events():
    class CustomEvents(EventEnum):
        TEST_EVENT = "test_event"

    # Dummy engine
    engine = Engine(lambda engine, batch: 0)
    engine.register_events(*CustomEvents)
    engine.register_events("a", "b", "c")

    evs = [CustomEvents.TEST_EVENT, "a", "b", "c"]

    # Handle is never called
    handlers = [(e, MagicMock()) for e in evs]
    for e, h in handlers:
        engine.add_event_handler(e, h)
    engine.run(range(1))
    for _, h in handlers:
        assert not h.called

    # Advanced engine
    def process_func(engine, batch):
        for e, _ in handlers:
            engine.fire_event(e)

    engine = Engine(process_func)
    engine.register_events(*CustomEvents)
    engine.register_events("a", "b", "c")

    # Handle should be called
    handlers = [(e, MagicMock()) for e in evs]
    for e, h in handlers:
        engine.add_event_handler(e, h)
    engine.run(range(1))
    for _, h in handlers:
        assert h.called


def test_has_registered_events_custom():
    """Test has_registered_events with custom events."""

    class TestEvents(EventEnum):
        CUSTOM_EVENT = "custom_event"

    engine = Engine(lambda e, b: None)

    # Custom event not registered yet
    assert not engine.has_registered_events(TestEvents.CUSTOM_EVENT)

    # Register custom event
    engine.register_events(TestEvents.CUSTOM_EVENT)

    # Now should return True
    assert engine.has_registered_events(TestEvents.CUSTOM_EVENT)


def test_custom_events_asserts():
    # Dummy engine
    engine = Engine(lambda engine, batch: 0)

    class A:
        pass

    with pytest.raises(TypeError, match=r"Value at \d of event_names should be a str or EventEnum"):
        engine.register_events(None)

    with pytest.raises(TypeError, match=r"Value at \d of event_names should be a str or EventEnum"):
        engine.register_events("str", None)

    with pytest.raises(TypeError, match=r"Value at \d of event_names should be a str or EventEnum"):
        engine.register_events(1)

    with pytest.raises(TypeError, match=r"Value at \d of event_names should be a str or EventEnum"):
        engine.register_events(A())

    assert Events.EPOCH_COMPLETED != 1
    assert Events.EPOCH_COMPLETED != "abc"
    assert Events.ITERATION_COMPLETED != Events.EPOCH_COMPLETED
    assert Events.ITERATION_COMPLETED != Events.EPOCH_COMPLETED(every=2)
    # In current implementation, EPOCH_COMPLETED and EPOCH_COMPLETED with event filter are the same
    assert Events.EPOCH_COMPLETED == Events.EPOCH_COMPLETED(every=2)
    assert Events.ITERATION_COMPLETED == Events.ITERATION_COMPLETED(every=2)


def test_custom_events_with_event_to_attr():
    class CustomEvents(EventEnum):
        TEST_EVENT = "test_event"

    custom_event_to_attr = {CustomEvents.TEST_EVENT: "test_event"}

    # Dummy engine
    engine = Engine(lambda engine, batch: 0)
    engine.register_events(*CustomEvents, event_to_attr=custom_event_to_attr)

    # Handle is never called
    handle = MagicMock()
    engine.add_event_handler(CustomEvents.TEST_EVENT, handle)
    engine.run(range(1))
    assert hasattr(engine.state, "test_event")
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

    custom_event_to_attr = "a"
    engine = Engine(lambda engine, batch: 0)
    with pytest.raises(ValueError):
        engine.register_events(*CustomEvents, event_to_attr=custom_event_to_attr)


def test_custom_events_with_events_list():
    class CustomEvents(EventEnum):
        TEST_EVENT = "test_event"

    def process_func(engine, batch):
        engine.fire_event(CustomEvents.TEST_EVENT)

    engine = Engine(process_func)
    engine.register_events(*CustomEvents)

    # Handle should be called
    handle = MagicMock()
    engine.add_event_handler(CustomEvents.TEST_EVENT | Events.STARTED, handle)
    engine.run(range(1))
    assert handle.called


def test_callable_events_with_wrong_inputs():
    def ef(e, i):
        return 1

    expected_raise = {
        # event_filter, every, once, before, after
        (None, None, None, None, None): True,  # raises ValueError
        (ef, None, None, None, None): False,
        (None, 2, None, None, None): False,
        (ef, 2, None, None, None): True,
        (None, None, 2, None, None): False,
        (ef, None, 2, None, None): True,
        (None, 2, 2, None, None): True,
        (ef, 2, 2, None, None): True,
        (None, None, None, 30, None): False,
        (ef, None, None, 30, None): True,
        (None, 2, None, 30, None): False,
        (ef, 2, None, 30, None): True,
        (None, None, 2, 30, None): True,
        (ef, None, 2, 30, None): True,
        (None, 2, 2, 30, None): True,
        (ef, 2, 2, 30, None): True,
        # event_filter, every, once, before, after
        (None, None, None, None, 10): False,
        (ef, None, None, None, 10): True,
        (None, 2, None, None, 10): False,
        (ef, 2, None, None, 10): True,
        (None, None, 2, None, 10): True,
        (ef, None, 2, None, 10): True,
        (None, 2, 2, None, 10): True,
        (ef, 2, 2, None, 10): True,
        (None, None, None, 25, 8): False,
        (ef, None, None, 25, 8): True,
        (None, 2, None, 25, 8): False,
        (ef, 2, None, 25, 8): True,
        (None, None, 2, 25, 8): True,
        (ef, None, 2, 25, 8): True,
        (None, 2, 2, 25, 8): True,
        (ef, 2, 2, 25, 8): True,
    }
    for event_filter in [None, ef]:
        for every in [None, 2]:
            for once in [None, 2]:
                for before, after in [(None, None), (None, 10), (30, None), (25, 8)]:
                    if expected_raise[(event_filter, every, once, before, after)]:
                        with pytest.raises(
                            ValueError,
                            match=r"Only one of the input arguments should be specified, "
                            "except before, after and every",
                        ):
                            Events.ITERATION_STARTED(
                                event_filter=event_filter, once=once, every=every, before=before, after=after
                            )
                    else:
                        Events.ITERATION_STARTED(
                            event_filter=event_filter, once=once, every=every, before=before, after=after
                        )

    with pytest.raises(TypeError, match=r"Argument event_filter should be a callable"):
        Events.ITERATION_STARTED(event_filter="123")

    with pytest.raises(ValueError, match=r"Argument every should be integer and greater than zero"):
        Events.ITERATION_STARTED(every=-1)

    with pytest.raises(
        ValueError, match=r"Argument once should either be a positive integer or a list of positive integers, got .+"
    ):
        Events.ITERATION_STARTED(once=-1)

    with pytest.raises(
        ValueError, match=r"Argument once should either be a positive integer or a list of positive integers, got .+"
    ):
        Events.ITERATION_STARTED(once=[1, 10.0, "pytorch"])

    with pytest.raises(
        ValueError, match=r"Argument once should either be a positive integer or a list of positive integers, got .+"
    ):
        Events.ITERATION_STARTED(once=[])

    with pytest.raises(ValueError, match=r"Argument before should be integer and greater or equal to zero"):
        Events.ITERATION_STARTED(before=-1)

    with pytest.raises(ValueError, match=r"Argument after should be integer and greater or equal to zero"):
        Events.ITERATION_STARTED(after=-1)

    with pytest.raises(ValueError, match=r"but will be called with"):
        Events.ITERATION_STARTED(event_filter=lambda x: x)

    with pytest.warns(UserWarning, match=r"default_event_filter is deprecated and will be removed"):
        Events.default_event_filter(None, None)


@pytest.mark.parametrize(
    "event",
    [
        Events.ITERATION_STARTED,
        Events.ITERATION_COMPLETED,
        Events.EPOCH_STARTED,
        Events.EPOCH_COMPLETED,
        Events.GET_BATCH_STARTED,
        Events.GET_BATCH_COMPLETED,
        Events.STARTED,
        Events.COMPLETED,
    ],
)
def test_callable_events(event):
    assert isinstance(event.value, str)

    def foo(engine, _):
        return True

    ret = event(event_filter=foo)
    assert isinstance(ret, CallableEventWithFilter)
    assert ret == event
    assert ret.filter == foo
    assert event.name in f"{ret}"

    ret = event(every=10)
    assert isinstance(ret, CallableEventWithFilter)
    assert ret == event
    assert ret.filter is not None
    assert event.name in f"{ret}"

    ret = event(once=10)
    assert isinstance(ret, CallableEventWithFilter)
    assert ret == event
    assert ret.filter is not None
    assert event.name in f"{ret}"

    ret = event(once=[1, 10])
    assert isinstance(ret, CallableEventWithFilter)
    assert ret == event
    assert ret.filter is not None
    assert event.name in f"{ret}"

    ret = event
    assert isinstance(ret, CallableEventWithFilter)
    assert ret.filter is None
    assert event.name in f"{ret}"


def test_callable_events_every_eq_one():
    e = Events.ITERATION_STARTED(every=1)
    assert isinstance(e, CallableEventWithFilter)


def test_has_handler_on_callable_events():
    engine = Engine(lambda e, b: 1)

    def foo(e):
        pass

    assert not engine.has_event_handler(foo)

    engine.add_event_handler(Events.EPOCH_STARTED, foo)
    assert engine.has_event_handler(foo)

    def bar(e):
        pass

    engine.add_event_handler(Events.EPOCH_COMPLETED(every=3), bar)
    assert engine.has_event_handler(bar)
    assert engine.has_event_handler(bar, Events.EPOCH_COMPLETED)
    assert engine.has_event_handler(bar, Events.EPOCH_COMPLETED(every=3))


def test_remove_event_handler_on_callable_events():
    engine = Engine(lambda e, b: 1)

    def foo(e):
        pass

    assert not engine.has_event_handler(foo)

    engine.add_event_handler(Events.EPOCH_STARTED, foo)
    assert engine.has_event_handler(foo)
    engine.remove_event_handler(foo, Events.EPOCH_STARTED)
    assert not engine.has_event_handler(foo)

    def bar(e):
        pass

    engine.add_event_handler(Events.EPOCH_COMPLETED(every=3), bar)
    assert engine.has_event_handler(bar)
    engine.remove_event_handler(bar, Events.EPOCH_COMPLETED)
    assert not engine.has_event_handler(bar)

    engine.add_event_handler(Events.EPOCH_COMPLETED(every=3), bar)
    assert engine.has_event_handler(bar)
    engine.remove_event_handler(bar, Events.EPOCH_COMPLETED(every=3))
    assert not engine.has_event_handler(bar)


def _test_every_event_filter_with_engine(device="cpu"):
    data = torch.rand(100, 4, device=device)

    def _test(event_name, event_attr, every, true_num_calls):
        engine = Engine(lambda e, b: b)

        counter = [0]
        counter_every = [0]
        num_calls = [0]

        @engine.on(event_name(every=every))
        def assert_every(engine):
            counter_every[0] += every
            assert getattr(engine.state, event_attr) % every == 0
            assert counter_every[0] == getattr(engine.state, event_attr)
            num_calls[0] += 1

        @engine.on(event_name(every=every))
        def assert_every_no_engine():
            assert getattr(engine.state, event_attr) % every == 0
            assert counter_every[0] == getattr(engine.state, event_attr)

        @engine.on(event_name)
        def assert_(engine):
            counter[0] += 1
            assert getattr(engine.state, event_attr) == counter[0]

        @engine.on(event_name)
        def assert_no_engine():
            assert getattr(engine.state, event_attr) == counter[0]

        engine.run(data, max_epochs=5)

        assert num_calls[0] == true_num_calls

    _test(Events.ITERATION_STARTED, "iteration", 10, 100 * 5 // 10)
    _test(Events.ITERATION_COMPLETED, "iteration", 10, 100 * 5 // 10)
    _test(Events.EPOCH_STARTED, "epoch", 2, 5 // 2)
    _test(Events.EPOCH_COMPLETED, "epoch", 2, 5 // 2)


def test_every_event_filter_with_engine():
    _test_every_event_filter_with_engine()


@pytest.mark.parametrize(
    "event_name, event_attr, before, expect_calls",
    [
        (Events.ITERATION_COMPLETED, "iteration", 0, 0),
        (Events.ITERATION_COMPLETED, "iteration", 300, 299),
        (Events.ITERATION_COMPLETED, "iteration", 501, 500),
        (Events.EPOCH_COMPLETED, "epoch", 0, 0),
        (Events.EPOCH_COMPLETED, "epoch", 3, 2),
        (Events.EPOCH_COMPLETED, "epoch", 6, 5),
    ],
)
def test_before_event_filter_with_engine(event_name, event_attr, before, expect_calls):
    data = range(100)

    engine = Engine(lambda e, b: 1)
    num_calls = 0

    @engine.on(event_name(before=before))
    def _before_event():
        nonlocal num_calls
        num_calls += 1
        assert getattr(engine.state, event_attr) < before

    engine.run(data, max_epochs=5)
    assert num_calls == expect_calls


@pytest.mark.parametrize(
    "event_name, event_attr, after, expect_calls",
    [
        (Events.ITERATION_STARTED, "iteration", 0, 500),
        (Events.ITERATION_COMPLETED, "iteration", 300, 200),
        (Events.ITERATION_COMPLETED, "iteration", 500, 0),
        (Events.EPOCH_STARTED, "epoch", 0, 5),
        (Events.EPOCH_COMPLETED, "epoch", 3, 2),
        (Events.EPOCH_COMPLETED, "epoch", 5, 0),
    ],
)
def test_after_event_filter_with_engine(event_name, event_attr, after, expect_calls):
    data = range(100)

    engine = Engine(lambda e, b: 1)
    num_calls = 0

    @engine.on(event_name(after=after))
    def _after_event():
        nonlocal num_calls
        num_calls += 1
        assert getattr(engine.state, event_attr) > after

    engine.run(data, max_epochs=5)
    assert num_calls == expect_calls


@pytest.mark.parametrize(
    "event_name, event_attr, before, after, expect_calls",
    [(Events.ITERATION_STARTED, "iteration", 300, 100, 199), (Events.EPOCH_COMPLETED, "epoch", 4, 1, 2)],
)
def test_before_and_after_event_filter_with_engine(event_name, event_attr, before, after, expect_calls):
    data = range(100)

    engine = Engine(lambda e, b: 1)
    num_calls = 0

    @engine.on(event_name(before=before, after=after))
    def _before_and_after_event():
        nonlocal num_calls
        num_calls += 1
        assert getattr(engine.state, event_attr) > after

    engine.run(data, max_epochs=5)
    assert num_calls == expect_calls


@pytest.mark.parametrize(
    "event_name, event_attr, every, before, after, expect_calls",
    [(Events.ITERATION_STARTED, "iteration", 5, 25, 8, 4), (Events.EPOCH_COMPLETED, "epoch", 2, 5, 1, 2)],
)
def test_every_before_and_after_event_filter_with_engine(event_name, event_attr, every, before, after, expect_calls):
    data = range(100)

    engine = Engine(lambda e, b: 1)
    num_calls = 0

    @engine.on(event_name(every=every, before=before, after=after))
    def _every_before_and_after_event():
        assert getattr(engine.state, event_attr) > after
        assert getattr(engine.state, event_attr) < before
        assert ((getattr(engine.state, event_attr) - after - 1) % every) == 0
        nonlocal num_calls
        num_calls += 1

    engine.run(data, max_epochs=5)
    assert num_calls == expect_calls


@pytest.mark.parametrize(
    "event_name, event_attr, once, expect_calls",
    [
        (Events.ITERATION_STARTED, "iteration", 2, 1),
        (Events.ITERATION_COMPLETED, "iteration", 2, 1),
        (Events.EPOCH_STARTED, "epoch", 2, 1),
        (Events.EPOCH_COMPLETED, "epoch", 2, 1),
        (Events.ITERATION_STARTED, "iteration", [1, 5], 2),
        (Events.ITERATION_COMPLETED, "iteration", [1, 5], 2),
        (Events.EPOCH_STARTED, "epoch", [1, 5], 2),
        (Events.EPOCH_COMPLETED, "epoch", [1, 5], 2),
    ],
)
def test_once_event_filter(event_name, event_attr, once, expect_calls):
    data = list(range(100))

    engine = Engine(lambda e, b: b)
    num_calls = [0]
    counter = [0]

    test_once = [once] if isinstance(once, int) else once

    @engine.on(event_name(once=once))
    def assert_once(engine):
        assert getattr(engine.state, event_attr) in test_once
        num_calls[0] += 1

    @engine.on(event_name)
    def assert_(engine):
        counter[0] += 1
        assert getattr(engine.state, event_attr) == counter[0]

    engine.run(data, max_epochs=10)
    assert num_calls[0] == expect_calls


def test_custom_event_filter_with_engine():
    special_events = [1, 2, 5, 7, 17, 20]

    def custom_event_filter(engine, event):
        if event in special_events:
            return True
        return False

    def _test(event_name, event_attr, true_num_calls):
        engine = Engine(lambda e, b: b)

        num_calls = [0]

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
    counter = [0]

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

    class CustomEvents2(EventEnum):
        TEST_EVENT = "test_event"

    CustomEvents2.TEST_EVENT(every=10)


def test_custom_callable_events_with_engine():
    class CustomEvents(EventEnum):
        TEST_EVENT = "test_event"

    event_to_attr = {CustomEvents.TEST_EVENT: "test_event"}

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

        num_calls = [0]

        @engine.on(event_name(event_filter=custom_event_filter))
        def assert_on_special_event(engine):
            assert getattr(engine.state, event_attr) == special_events.pop(0)
            num_calls[0] += 1

        d = list(range(50))
        engine.run(d, max_epochs=25)

        assert num_calls[0] == true_num_calls

    _test(CustomEvents.TEST_EVENT, "test_event", len(special_events))


def _test_every_event_filter_with_engine_with_dataloader(device):
    def _test(num_workers):
        max_epochs = 3
        batch_size = 4
        num_iters = 21
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))

        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory="cuda" in torch.device(device).type,
            drop_last=True,
            shuffle=True,
        )
        seen_batchs = []

        def update_fn(_, batch):
            batch_to_device = batch.to(device)
            seen_batchs.append(batch)

        engine = Engine(update_fn)

        def foo(_):
            pass

        engine.add_event_handler(Events.EPOCH_STARTED(every=2), foo)
        engine.run(dataloader, max_epochs=max_epochs)
        engine = None

        import gc

        gc.collect()
        assert len(gc.garbage) == 0

    _test(num_workers=0)
    _test(num_workers=1)


def test_every_event_filter_with_engine_with_dataloader():
    _test_every_event_filter_with_engine_with_dataloader("cpu")


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_every_event_filter_with_engine(device)
    _test_every_event_filter_with_engine_with_dataloader(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_every_event_filter_with_engine(device)
    _test_every_event_filter_with_engine_with_dataloader(device)


def test_event_list():
    e1 = Events.ITERATION_STARTED(once=1)
    e2 = Events.ITERATION_STARTED(every=3)
    e3 = Events.COMPLETED

    event_list = e1 | e2 | e3

    assert isinstance(event_list, EventsList)
    assert len(event_list) == 3
    assert event_list[0] == e1
    assert event_list[1] == e2
    assert event_list[2] == e3


def test_list_of_events():
    def _test(event_list, true_iterations):
        engine = Engine(lambda e, b: b)

        iterations = []

        num_calls = [0]

        @engine.on(event_list)
        def execute_some_handler(e):
            iterations.append(e.state.iteration)
            num_calls[0] += 1

        engine.run(range(3), max_epochs=5)

        assert iterations == true_iterations
        assert num_calls[0] == len(true_iterations)

    _test(Events.ITERATION_STARTED(once=1) | Events.ITERATION_STARTED(once=1), [1, 1])
    _test(Events.ITERATION_STARTED(once=1) | Events.ITERATION_STARTED(once=10), [1, 10])
    _test(Events.ITERATION_STARTED(once=1) | Events.ITERATION_STARTED(every=3), [1, 3, 6, 9, 12, 15])
    _test(Events.ITERATION_STARTED(once=8) | Events.ITERATION_STARTED(before=3), [1, 2, 8])
    _test(Events.ITERATION_STARTED(once=1) | Events.ITERATION_STARTED(after=12), [1, 13, 14, 15])
