import functools
import gc
from unittest.mock import call, create_autospec, MagicMock

import pytest
from pytest import raises

from ignite.engine import Engine, Events, State
from ignite.engine.events import EventsList


class DummyEngine(Engine):
    def __init__(self):
        super(DummyEngine, self).__init__(lambda e, b: 1)

    def run(self, num_times):
        self.state = State()
        for _ in range(num_times):
            self.fire_event(Events.STARTED)
            self.fire_event(Events.COMPLETED)
        return self.state


def test_add_event_handler_raises_with_invalid_event():
    engine = Engine(lambda e, b: 1)

    with pytest.raises(ValueError, match=r"is not a valid event for this Engine"):
        engine.add_event_handler("incorrect", lambda engine: None)


def test_add_event_handler_raises_with_invalid_signature():
    engine = Engine(MagicMock())

    def handler(engine):
        pass

    engine.add_event_handler(Events.STARTED, handler)
    engine.add_event_handler(Events.STARTED, handler, 1)

    def handler_with_args(engine, a):
        pass

    engine.add_event_handler(Events.STARTED, handler_with_args, 1)
    with pytest.raises(ValueError):
        engine.add_event_handler(Events.STARTED, handler_with_args)

    def handler_with_kwargs(engine, b=42):
        pass

    engine.add_event_handler(Events.STARTED, handler_with_kwargs, b=2)
    with pytest.raises(ValueError):
        engine.add_event_handler(Events.STARTED, handler_with_kwargs, c=3)
    engine.add_event_handler(Events.STARTED, handler_with_kwargs, 1, b=2)

    def handler_with_args_and_kwargs(engine, a, b=42):
        pass

    engine.add_event_handler(Events.STARTED, handler_with_args_and_kwargs, 1, b=2)
    engine.add_event_handler(Events.STARTED, handler_with_args_and_kwargs, 1, 2, b=2)
    with pytest.raises(ValueError):
        engine.add_event_handler(Events.STARTED, handler_with_args_and_kwargs, 1, b=2, c=3)


def test_add_event_handler():
    engine = DummyEngine()

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    started_counter = Counter()

    def handle_iteration_started(engine, counter):
        counter.count += 1

    engine.add_event_handler(Events.STARTED, handle_iteration_started, started_counter)

    completed_counter = Counter()

    def handle_iteration_completed(engine, counter):
        counter.count += 1

    engine.add_event_handler(Events.COMPLETED, handle_iteration_completed, completed_counter)

    engine.run(15)

    assert started_counter.count == 15
    assert completed_counter.count == 15


def test_add_event_handler_without_engine():
    engine = DummyEngine()

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    started_counter = Counter()

    def handle_iteration_started():
        started_counter.count += 1

    engine.add_event_handler(Events.STARTED, handle_iteration_started)

    completed_counter = Counter()

    def handle_iteration_completed(counter):
        counter.count += 1

    engine.add_event_handler(Events.COMPLETED, handle_iteration_completed, completed_counter)

    engine.run(15)

    assert started_counter.count == 15
    assert completed_counter.count == 15


def test_adding_multiple_event_handlers():
    mock_fn_1 = create_autospec(spec=lambda x: None)
    mock_fn_2 = create_autospec(spec=lambda x: None)

    engine = DummyEngine()
    handlers = [mock_fn_1, mock_fn_2]
    for handler in handlers:
        engine.add_event_handler(Events.STARTED, handler)

    engine.run(1)
    for handler in handlers:
        handler.assert_called_once_with(engine)


@pytest.mark.parametrize(
    "event1, event2",
    [
        (Events.STARTED, Events.COMPLETED),
        (Events.EPOCH_STARTED, Events.EPOCH_COMPLETED),
        (Events.ITERATION_STARTED, Events.ITERATION_COMPLETED),
        (Events.ITERATION_STARTED(every=2), Events.ITERATION_COMPLETED(every=2)),
    ],
)
def test_event_removable_handle(event1, event2):
    # Removable handle removes event from engine.
    engine = Engine(lambda e, b: None)
    handler = create_autospec(spec=lambda x: None)
    assert not hasattr(handler, "_parent")

    removable_handle = engine.add_event_handler(event1, handler)
    assert engine.has_event_handler(handler, event1)

    engine.run([1, 2])
    handler.assert_any_call(engine)
    num_calls = handler.call_count

    removable_handle.remove()
    assert not engine.has_event_handler(handler, event1)

    # Second engine pass does not fire handle again.
    engine.run([1, 2])
    # Assert that handler wasn't call
    assert handler.call_count == num_calls

    # Removable handle can be used as a context manager
    handler = create_autospec(spec=lambda x: None)

    with engine.add_event_handler(event1, handler):
        assert engine.has_event_handler(handler, event1)
        engine.run([1, 2])

    assert not engine.has_event_handler(handler, event1)
    handler.assert_any_call(engine)
    num_calls = handler.call_count

    engine.run([1, 2])
    # Assert that handler wasn't call
    assert handler.call_count == num_calls

    # Removeable handle only effects a single event registration
    handler = MagicMock(spec_set=True)

    with engine.add_event_handler(event1, handler):
        with engine.add_event_handler(event2, handler):
            assert engine.has_event_handler(handler, event1)
            assert engine.has_event_handler(handler, event2)
        assert engine.has_event_handler(handler, event1)
        assert not engine.has_event_handler(handler, event2)
    assert not engine.has_event_handler(handler, event1)
    assert not engine.has_event_handler(handler, event2)

    # Removeable handle is re-enter and re-exitable

    handler = MagicMock(spec_set=True)

    remove = engine.add_event_handler(event1, handler)

    with remove:
        with remove:
            assert engine.has_event_handler(handler, event1)
        assert not engine.has_event_handler(handler, event1)
    assert not engine.has_event_handler(handler, event1)

    # Removeable handle is a weakref, does not keep engine or event alive
    def _add_in_closure():
        _engine = Engine(lambda e, b: None)

        def _handler(_):
            pass

        _handle = _engine.add_event_handler(event1, _handler)
        assert _handle.engine() is _engine

        if event1.filter is None:
            assert _handle.handler() is _handler
        else:
            assert _handle.handler()._parent() is _handler

        return _handle

    removable_handle = _add_in_closure()

    # gc.collect, resolving reference cycles in engine/state
    # required to ensure object deletion in python2
    gc.collect()

    assert removable_handle.engine() is None
    assert removable_handle.handler() is None


def test_events_list_removable_handle():
    # Removable handle removes event from engine.
    engine = DummyEngine()
    handler = create_autospec(spec=lambda x: None)
    assert not hasattr(handler, "_parent")

    events_list = Events.STARTED | Events.COMPLETED

    removable_handle = engine.add_event_handler(events_list, handler)
    for e in events_list:
        assert engine.has_event_handler(handler, e)

    engine.run(1)
    calls = [call(engine), call(engine)]
    handler.assert_has_calls(calls)
    assert handler.call_count == 2

    removable_handle.remove()
    for e in events_list:
        assert not engine.has_event_handler(handler, e)

    # Second engine pass does not fire handle again.
    engine.run(1)
    handler.assert_has_calls(calls)
    assert handler.call_count == 2

    # Removable handle can be used as a context manager
    handler = create_autospec(spec=lambda x: None)

    with engine.add_event_handler(events_list, handler):
        for e in events_list:
            assert engine.has_event_handler(handler, e)
        engine.run(1)

    for e in events_list:
        assert not engine.has_event_handler(handler, e)
    handler.assert_has_calls(calls)
    assert handler.call_count == 2

    engine.run(1)
    handler.assert_has_calls(calls)
    assert handler.call_count == 2

    # Removeable handle only effects a single event registration
    handler = create_autospec(spec=lambda x: None)

    other_events_list = Events.EPOCH_STARTED | Events.EPOCH_COMPLETED

    with engine.add_event_handler(events_list, handler):
        with engine.add_event_handler(other_events_list, handler):
            for e in events_list:
                assert engine.has_event_handler(handler, e)
            for e in other_events_list:
                assert engine.has_event_handler(handler, e)
        for e in events_list:
            assert engine.has_event_handler(handler, e)
        for e in other_events_list:
            assert not engine.has_event_handler(handler, e)
    for e in events_list:
        assert not engine.has_event_handler(handler, e)
    for e in other_events_list:
        assert not engine.has_event_handler(handler, e)

    # Removeable handle is re-enter and re-exitable

    handler = create_autospec(spec=lambda x: None)

    remove = engine.add_event_handler(events_list, handler)

    with remove:
        with remove:
            for e in events_list:
                assert engine.has_event_handler(handler, e)
        for e in events_list:
            assert not engine.has_event_handler(handler, e)
    for e in events_list:
        assert not engine.has_event_handler(handler, e)

    # Removeable handle is a weakref, does not keep engine or event alive
    def _add_in_closure():
        _engine = DummyEngine()

        def _handler(_):
            pass

        _handle = _engine.add_event_handler(events_list, _handler)
        assert _handle.engine() is _engine
        assert _handle.handler() is _handler

        return _handle

    removable_handle = _add_in_closure()

    # gc.collect, resolving reference cycles in engine/state
    # required to ensure object deletion in python2
    gc.collect()

    assert removable_handle.engine() is None
    assert removable_handle.handler() is None


def test_eventslist__append_raises():
    ev_list = EventsList()
    with pytest.raises(TypeError, match=r"Argument event should be Events or CallableEventWithFilter"):
        ev_list._append("abc")


def test_has_event_handler():
    engine = DummyEngine()
    handlers = [MagicMock(spec_set=True), MagicMock(spec_set=True)]
    m = MagicMock(spec_set=True)
    for handler in handlers:
        engine.add_event_handler(Events.STARTED, handler)
    engine.add_event_handler(Events.COMPLETED, m)

    for handler in handlers:
        assert engine.has_event_handler(handler, Events.STARTED)
        assert engine.has_event_handler(handler)
        assert not engine.has_event_handler(handler, Events.COMPLETED)
        assert not engine.has_event_handler(handler, Events.EPOCH_STARTED)

    assert not engine.has_event_handler(m, Events.STARTED)
    assert engine.has_event_handler(m, Events.COMPLETED)
    assert engine.has_event_handler(m)
    assert not engine.has_event_handler(m, Events.EPOCH_STARTED)


def test_remove_event_handler():
    engine = DummyEngine()

    with pytest.raises(ValueError, match=r"Input event name"):
        engine.remove_event_handler(lambda x: x, "an event")

    def on_started(engine):
        return 0

    engine.add_event_handler(Events.STARTED, on_started)

    with pytest.raises(ValueError, match=r"Input handler"):
        engine.remove_event_handler(lambda x: x, Events.STARTED)

    h1 = MagicMock(spec_set=True)
    h2 = MagicMock(spec_set=True)
    handlers = [h1, h2]
    m = MagicMock(spec_set=True)
    for handler in handlers:
        engine.add_event_handler(Events.EPOCH_STARTED, handler)
    engine.add_event_handler(Events.EPOCH_COMPLETED, m)

    assert len(engine._event_handlers[Events.EPOCH_STARTED]) == 2
    engine.remove_event_handler(h1, Events.EPOCH_STARTED)
    assert len(engine._event_handlers[Events.EPOCH_STARTED]) == 1
    assert engine._event_handlers[Events.EPOCH_STARTED][0][0] == h2

    assert len(engine._event_handlers[Events.EPOCH_COMPLETED]) == 1
    engine.remove_event_handler(m, Events.EPOCH_COMPLETED)
    assert len(engine._event_handlers[Events.EPOCH_COMPLETED]) == 0


def test_args_and_kwargs_are_passed_to_event():
    engine = DummyEngine()
    kwargs = {"a": "a", "b": "b"}
    args = (1, 2, 3)
    handlers = []
    for event in [Events.STARTED, Events.COMPLETED]:
        handler = create_autospec(spec=lambda e, x1, x2, x3, a, b: None)
        engine.add_event_handler(event, handler, *args, **kwargs)
        handlers.append(handler)

    engine.run(1)
    called_handlers = [handle for handle in handlers if handle.called]
    assert len(called_handlers) == 2

    for handler in called_handlers:
        handler_args, handler_kwargs = handler.call_args
        assert handler_args[0] == engine
        assert handler_args[1::] == args
        assert handler_kwargs == kwargs


def test_on_decorator_raises_with_invalid_event():
    engine = DummyEngine()
    with pytest.raises(ValueError):

        @engine.on("incorrect")
        def f(engine):
            pass


def test_on_decorator():
    engine = DummyEngine()

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    started_counter = Counter()

    @engine.on(Events.STARTED, started_counter)
    def handle_iteration_started(engine, started_counter):
        started_counter.count += 1

    completed_counter = Counter()

    @engine.on(Events.COMPLETED, completed_counter)
    def handle_iteration_completed(engine, completed_counter):
        completed_counter.count += 1

    engine.run(15)

    assert started_counter.count == 15
    assert completed_counter.count == 15


def test_returns_state():
    engine = Engine(MagicMock(return_value=1))
    state = engine.run([0])

    assert isinstance(state, State)


def test_state_attributes():
    dataloader = [1, 2, 3]
    engine = Engine(MagicMock(return_value=1))
    state = engine.run(dataloader, max_epochs=3)

    assert state.iteration == 9
    assert state.output == 1
    assert state.batch == 3
    assert state.dataloader == dataloader
    assert state.epoch == 3
    assert state.max_epochs == 3
    assert state.metrics == {}

    with pytest.raises(RuntimeError, match=r"Unknown event name"):
        state.get_event_attrib_value("abc")


def test_default_exception_handler():
    update_function = MagicMock(side_effect=ValueError())
    engine = Engine(update_function)

    with raises(ValueError):
        engine.run([1])


def test_custom_exception_handler():
    value_error = ValueError()
    update_function = MagicMock(side_effect=value_error)

    engine = Engine(update_function)

    class ExceptionCounter(object):
        def __init__(self):
            self.exceptions = []

        def __call__(self, engine, e):
            self.exceptions.append(e)

    counter = ExceptionCounter()
    engine.add_event_handler(Events.EXCEPTION_RAISED, counter)
    engine.run([1])

    # only one call from _run_once_over_data, since the exception is swallowed
    assert len(counter.exceptions) == 1 and counter.exceptions[0] == value_error


def test_event_handlers_with_decoration():
    engine = Engine(lambda e, b: b)

    def decorated(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return fun(*args, **kwargs)

        return wrapper

    values = []

    def foo():
        values.append("foo")

    @decorated
    def decorated_foo():
        values.append("decorated_foo")

    engine.add_event_handler(Events.EPOCH_STARTED, foo)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), foo)
    engine.add_event_handler(Events.EPOCH_STARTED, decorated_foo)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), decorated_foo)

    def foo_args(e):
        values.append("foo_args")
        values.append(e.state.iteration)

    @decorated
    def decorated_foo_args(e):
        values.append("decorated_foo_args")
        values.append(e.state.iteration)

    engine.add_event_handler(Events.EPOCH_STARTED, foo_args)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), foo_args)
    engine.add_event_handler(Events.EPOCH_STARTED, decorated_foo_args)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), decorated_foo_args)

    class Foo:
        def __init__(self):
            self.values = []

        def foo(self):
            self.values.append("foo")

        @decorated
        def decorated_foo(self):
            self.values.append("decorated_foo")

        def foo_args(self, e):
            self.values.append("foo_args")
            self.values.append(e.state.iteration)

        @decorated
        def decorated_foo_args(self, e):
            self.values.append("decorated_foo_args")
            self.values.append(e.state.iteration)

    foo = Foo()

    engine.add_event_handler(Events.EPOCH_STARTED, foo.foo)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), foo.foo)
    engine.add_event_handler(Events.EPOCH_STARTED, foo.decorated_foo)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), foo.decorated_foo)
    engine.add_event_handler(Events.EPOCH_STARTED, foo.foo_args)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), foo.foo_args)
    engine.add_event_handler(Events.EPOCH_STARTED, foo.decorated_foo_args)
    engine.add_event_handler(Events.EPOCH_STARTED(every=2), foo.decorated_foo_args)

    engine.run([0], max_epochs=2)

    assert values == foo.values
