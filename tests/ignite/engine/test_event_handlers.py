import gc

from unittest.mock import MagicMock

import pytest
from pytest import raises

from ignite.engine import Engine, Events, State


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
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
        engine.add_event_handler(Events.STARTED, handler_with_kwargs, 1, b=2)

    def handler_with_args_and_kwargs(engine, a, b=42):
        pass

    engine.add_event_handler(Events.STARTED, handler_with_args_and_kwargs, 1, b=2)
    with pytest.raises(ValueError):
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


def test_adding_multiple_event_handlers():
    engine = DummyEngine()
    handlers = [MagicMock(), MagicMock()]
    for handler in handlers:
        engine.add_event_handler(Events.STARTED, handler)

    engine.run(1)
    for handler in handlers:
        handler.assert_called_once_with(engine)


def test_event_removable_handle():

    # Removable handle removes event from engine.
    engine = DummyEngine()
    handler = MagicMock(spec_set=True)
    assert not hasattr(handler, "_parent")

    removable_handle = engine.add_event_handler(Events.STARTED, handler)
    assert engine.has_event_handler(handler, Events.STARTED)

    engine.run(1)
    handler.assert_called_once_with(engine)

    removable_handle.remove()
    assert not engine.has_event_handler(handler, Events.STARTED)

    # Second engine pass does not fire handle again.
    engine.run(1)
    handler.assert_called_once_with(engine)

    # Removable handle can be used as a context manager
    handler = MagicMock(spec_set=True)

    with engine.add_event_handler(Events.STARTED, handler):
        assert engine.has_event_handler(handler, Events.STARTED)
        engine.run(1)

    assert not engine.has_event_handler(handler, Events.STARTED)
    handler.assert_called_once_with(engine)

    engine.run(1)
    handler.assert_called_once_with(engine)

    # Removeable handle only effects a single event registration
    handler = MagicMock(spec_set=True)

    with engine.add_event_handler(Events.STARTED, handler):
        with engine.add_event_handler(Events.COMPLETED, handler):
            assert engine.has_event_handler(handler, Events.STARTED)
            assert engine.has_event_handler(handler, Events.COMPLETED)
        assert engine.has_event_handler(handler, Events.STARTED)
        assert not engine.has_event_handler(handler, Events.COMPLETED)
    assert not engine.has_event_handler(handler, Events.STARTED)
    assert not engine.has_event_handler(handler, Events.COMPLETED)

    # Removeable handle is re-enter and re-exitable

    handler = MagicMock(spec_set=True)

    remove = engine.add_event_handler(Events.STARTED, handler)

    with remove:
        with remove:
            assert engine.has_event_handler(handler, Events.STARTED)
        assert not engine.has_event_handler(handler, Events.STARTED)
    assert not engine.has_event_handler(handler, Events.STARTED)

    # Removeable handle is a weakref, does not keep engine or event alive
    def _add_in_closure():
        _engine = DummyEngine()

        def _handler(_):
            pass

        _handle = _engine.add_event_handler(Events.STARTED, _handler)
        assert _handle.engine() is _engine
        assert _handle.handler() is _handler

        return _handle

    removable_handle = _add_in_closure()

    # gc.collect, resolving reference cycles in engine/state
    # required to ensure object deletion in python2
    gc.collect()

    assert removable_handle.engine() is None
    assert removable_handle.handler() is None


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
        handler = MagicMock(spec_set=True)
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
    state = engine.run([0,])

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
