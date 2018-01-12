from enum import Enum

import pytest
from mock import MagicMock

from ignite.engine import Engine


class Events(Enum):
    START = "start"
    END = "end"


class DummyEngine(Engine):
    def __init__(self):
        super(DummyEngine, self).__init__(Events)

    def run(self, num_times):
        self._logger.error("hello world")
        for _ in range(num_times):
            self._fire_event(Events.START)
            self._fire_event(Events.END)


def test_terminate():
    engine = DummyEngine()
    assert not engine.should_terminate
    engine.terminate()
    assert engine.should_terminate


def test_add_event_handler_raises_with_invalid_event():
    engine = DummyEngine()

    with pytest.raises(ValueError):
        engine.add_event_handler("incorrect", lambda engine: None)


def test_add_event_handler():
    engine = DummyEngine()

    class Counter(object):
        def __init__(self, count=0):
            self.count = count

    started_counter = Counter()

    def handle_training_iteration_started(engine, counter):
        counter.count += 1
    engine.add_event_handler(Events.START, handle_training_iteration_started, started_counter)

    completed_counter = Counter()

    def handle_training_iteration_completed(engine, counter):
        counter.count += 1
    engine.add_event_handler(Events.END, handle_training_iteration_completed, completed_counter)

    engine.run(15)

    assert started_counter.count == 15
    assert completed_counter.count == 15


def test_adding_multiple_event_handlers():
    engine = DummyEngine()
    handlers = [MagicMock(), MagicMock()]
    for handler in handlers:
        engine.add_event_handler(Events.START, handler)

    engine.run(1)
    for handler in handlers:
        handler.assert_called_once_with(engine)


def test_args_and_kwargs_are_passed_to_event():
    engine = DummyEngine()
    kwargs = {'a': 'a', 'b': 'b'}
    args = (1, 2, 3)
    handlers = []
    for event in Events:
        handler = MagicMock()
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

    @engine.on(Events.START, started_counter)
    def handle_training_iteration_started(engine, started_counter):
        started_counter.count += 1

    completed_counter = Counter()

    @engine.on(Events.END, completed_counter)
    def handle_training_iteration_completed(engine, completed_counter):
        completed_counter.count += 1

    engine.run(15)

    assert started_counter.count == 15
    assert completed_counter.count == 15
