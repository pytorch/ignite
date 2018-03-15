from enum import Enum

import pytest
from mock import MagicMock

from ignite.engines import Engine, Events, State


def process_func(batch):
    return 1


class DummyEngine(Engine):
    def __init__(self):
        super(DummyEngine, self).__init__(process_func)

    def run(self, num_times):
        state = State()
        for _ in range(num_times):
            self._fire_event(Events.STARTED, state)
            self._fire_event(Events.COMPLETED, state)
        return state


def test_terminate():
    engine = DummyEngine()
    assert not engine.should_terminate
    engine.terminate()
    assert engine.should_terminate


def test_add_event_handler_raises_with_invalid_event():
    engine = DummyEngine()

    with pytest.raises(ValueError):
        engine.add_event_handler("incorrect", lambda engine, state: None)


def test_add_event_handler():
    engine = DummyEngine()

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    started_counter = Counter()

    def handle_training_iteration_started(engine, state, counter):
        counter.count += 1
    engine.add_event_handler(Events.STARTED, handle_training_iteration_started, started_counter)

    completed_counter = Counter()

    def handle_training_iteration_completed(engine, state, counter):
        counter.count += 1
    engine.add_event_handler(Events.COMPLETED, handle_training_iteration_completed, completed_counter)

    engine.run(15)

    assert started_counter.count == 15
    assert completed_counter.count == 15


def test_adding_multiple_event_handlers():
    engine = DummyEngine()
    handlers = [MagicMock(), MagicMock()]
    for handler in handlers:
        engine.add_event_handler(Events.STARTED, handler)

    state = engine.run(1)
    for handler in handlers:
        handler.assert_called_once_with(engine, state)


def test_args_and_kwargs_are_passed_to_event():
    engine = DummyEngine()
    kwargs = {'a': 'a', 'b': 'b'}
    args = (1, 2, 3)
    handlers = []
    for event in [Events.STARTED, Events.COMPLETED]:
        handler = MagicMock()
        engine.add_event_handler(event, handler, *args, **kwargs)
        handlers.append(handler)

    state = engine.run(1)
    called_handlers = [handle for handle in handlers if handle.called]
    assert len(called_handlers) == 2

    for handler in called_handlers:
        handler_args, handler_kwargs = handler.call_args
        assert handler_args[0] == engine
        assert handler_args[1] == state
        assert handler_args[2::] == args
        assert handler_kwargs == kwargs


def test_on_decorator_raises_with_invalid_event():
    engine = DummyEngine()
    with pytest.raises(ValueError):
        @engine.on("incorrect")
        def f(engine, state):
            pass


def test_on_decorator():
    engine = DummyEngine()

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    started_counter = Counter()

    @engine.on(Events.STARTED, started_counter)
    def handle_training_iteration_started(engine, state, started_counter):
        started_counter.count += 1

    completed_counter = Counter()

    @engine.on(Events.COMPLETED, completed_counter)
    def handle_training_iteration_completed(engine, state, completed_counter):
        completed_counter.count += 1

    engine.run(15)

    assert started_counter.count == 15
    assert completed_counter.count == 15
