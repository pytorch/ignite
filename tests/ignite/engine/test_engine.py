from __future__ import division
from enum import Enum
import gc

import pytest
from mock import call, MagicMock, Mock
from pytest import raises, approx
import numpy as np
import torch
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD

from ignite.engine import Engine, Events, State, create_supervised_trainer, create_supervised_evaluator
from ignite.engine.engine import CallableEvents, EventWithFilter
from ignite.metrics import MeanSquaredError


def process_func(engine, batch):
    return 1


class DummyEngine(Engine):
    def __init__(self):
        super(DummyEngine, self).__init__(process_func)

    def run(self, num_times):
        self.state = State()
        for _ in range(num_times):
            self.fire_event(Events.STARTED)
            self.fire_event(Events.COMPLETED)
        return self.state


def test_terminate():
    engine = DummyEngine()
    assert not engine.should_terminate
    engine.terminate()
    assert engine.should_terminate


def test_invalid_process_raises_with_invalid_signature():
    with pytest.raises(ValueError):
        Engine(None)

    with pytest.raises(ValueError):
        Engine(lambda: None)

    with pytest.raises(ValueError):
        Engine(lambda batch: None)

    with pytest.raises(ValueError):
        Engine(lambda engine, batch, extra_arg: None)


def test_add_event_handler_raises_with_invalid_event():
    engine = DummyEngine()

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
    handler = MagicMock()

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
    handler = MagicMock()

    with engine.add_event_handler(Events.STARTED, handler):
        assert engine.has_event_handler(handler, Events.STARTED)
        engine.run(1)

    assert not engine.has_event_handler(handler, Events.STARTED)
    handler.assert_called_once_with(engine)

    engine.run(1)
    handler.assert_called_once_with(engine)

    # Removeable handle only effects a single event registration
    handler = MagicMock()

    with engine.add_event_handler(Events.STARTED, handler):
        with engine.add_event_handler(Events.COMPLETED, handler):
            assert engine.has_event_handler(handler, Events.STARTED)
            assert engine.has_event_handler(handler, Events.COMPLETED)
        assert engine.has_event_handler(handler, Events.STARTED)
        assert not engine.has_event_handler(handler, Events.COMPLETED)
    assert not engine.has_event_handler(handler, Events.STARTED)
    assert not engine.has_event_handler(handler, Events.COMPLETED)

    # Removeable handle is re-enter and re-exitable

    handler = MagicMock()

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
    handlers = [MagicMock(), MagicMock()]
    m = MagicMock()
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

    with pytest.raises(ValueError, match=r'Input event name'):
        engine.remove_event_handler(lambda x: x, "an event")

    def on_started(engine):
        return 0

    engine.add_event_handler(Events.STARTED, on_started)

    with pytest.raises(ValueError, match=r'Input handler'):
        engine.remove_event_handler(lambda x: x, Events.STARTED)

    h1 = MagicMock()
    h2 = MagicMock()
    handlers = [h1, h2]
    m = MagicMock()
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
    kwargs = {'a': 'a', 'b': 'b'}
    args = (1, 2, 3)
    handlers = []
    for event in [Events.STARTED, Events.COMPLETED]:
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

    with pytest.raises(ValueError, match=r"Argument every should be integer and greater than one"):
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
    state = engine.run([])

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


def test_current_epoch_counter_increases_every_epoch():
    engine = Engine(MagicMock(return_value=1))
    max_epochs = 5

    class EpochCounter(object):
        def __init__(self):
            self.current_epoch_count = 1

        def __call__(self, engine):
            assert engine.state.epoch == self.current_epoch_count
            self.current_epoch_count += 1

    engine.add_event_handler(Events.EPOCH_STARTED, EpochCounter())

    state = engine.run([1], max_epochs=max_epochs)

    assert state.epoch == max_epochs


def test_current_iteration_counter_increases_every_iteration():
    batches = [1, 2, 3]
    engine = Engine(MagicMock(return_value=1))
    max_epochs = 5

    class IterationCounter(object):
        def __init__(self):
            self.current_iteration_count = 1

        def __call__(self, engine):
            assert engine.state.iteration == self.current_iteration_count
            self.current_iteration_count += 1

    engine.add_event_handler(Events.ITERATION_STARTED, IterationCounter())

    state = engine.run(batches, max_epochs=max_epochs)

    assert state.iteration == max_epochs * len(batches)


def test_stopping_criterion_is_max_epochs():
    engine = Engine(MagicMock(return_value=1))
    max_epochs = 5
    state = engine.run([1], max_epochs=max_epochs)
    assert state.epoch == max_epochs


def test_terminate_at_end_of_epoch_stops_run():
    max_epochs = 5
    last_epoch_to_run = 3

    engine = Engine(MagicMock(return_value=1))

    def end_of_epoch_handler(engine):
        if engine.state.epoch == last_epoch_to_run:
            engine.terminate()

    engine.add_event_handler(Events.EPOCH_COMPLETED, end_of_epoch_handler)

    assert not engine.should_terminate

    state = engine.run([1], max_epochs=max_epochs)

    assert state.epoch == last_epoch_to_run
    assert engine.should_terminate


def test_terminate_at_start_of_epoch_stops_run_after_completing_iteration():
    max_epochs = 5
    epoch_to_terminate_on = 3
    batches_per_epoch = [1, 2, 3]

    engine = Engine(MagicMock(return_value=1))

    def start_of_epoch_handler(engine):
        if engine.state.epoch == epoch_to_terminate_on:
            engine.terminate()

    engine.add_event_handler(Events.EPOCH_STARTED, start_of_epoch_handler)

    assert not engine.should_terminate

    state = engine.run(batches_per_epoch, max_epochs=max_epochs)

    # epoch is not completed so counter is not incremented
    assert state.epoch == epoch_to_terminate_on
    assert engine.should_terminate
    # completes first iteration
    assert state.iteration == ((epoch_to_terminate_on - 1) * len(batches_per_epoch)) + 1


def test_terminate_stops_run_mid_epoch():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch + 3
    engine = Engine(MagicMock(return_value=1))

    def start_of_iteration_handler(engine):
        if engine.state.iteration == iteration_to_stop:
            engine.terminate()

    engine.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    state = engine.run(data=[None] * num_iterations_per_epoch, max_epochs=3)
    # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
    assert (state.iteration == iteration_to_stop)
    assert state.epoch == np.ceil(iteration_to_stop / num_iterations_per_epoch)  # it starts from 0


def test_terminate_epoch_stops_mid_epoch():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch + 3
    engine = Engine(MagicMock(return_value=1))

    def start_of_iteration_handler(engine):
        if engine.state.iteration == iteration_to_stop:
            engine.terminate_epoch()

    max_epochs = 3
    engine.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    state = engine.run(data=[None] * num_iterations_per_epoch, max_epochs=max_epochs)
    # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
    assert state.iteration == num_iterations_per_epoch * (max_epochs - 1) + \
        iteration_to_stop % num_iterations_per_epoch


def _create_mock_data_loader(epochs, batches_per_epoch):
    batches = [MagicMock()] * batches_per_epoch
    data_loader_manager = MagicMock()
    batch_iterators = [iter(batches) for _ in range(epochs)]

    data_loader_manager.__iter__.side_effect = batch_iterators

    return data_loader_manager


def test_iteration_events_are_fired():
    max_epochs = 5
    num_batches = 3
    data = _create_mock_data_loader(max_epochs, num_batches)

    engine = Engine(MagicMock(return_value=1))

    mock_manager = Mock()
    iteration_started = Mock()
    engine.add_event_handler(Events.ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    engine.add_event_handler(Events.ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    engine.run(data, max_epochs=max_epochs)

    assert iteration_started.call_count == num_batches * max_epochs
    assert iteration_complete.call_count == num_batches * max_epochs

    expected_calls = []
    for i in range(max_epochs * num_batches):
        expected_calls.append(call.iteration_started(engine))
        expected_calls.append(call.iteration_complete(engine))

    assert mock_manager.mock_calls == expected_calls


def test_create_supervised_trainer():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    trainer = create_supervised_trainer(model, optimizer, mse_loss)

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert model.weight.data[0, 0].item() == approx(1.3)
    assert model.bias.item() == approx(0.8)


def test_create_supervised_trainer_with_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    trainer = create_supervised_trainer(model, optimizer, mse_loss, device='cpu')

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert model.weight.data[0, 0].item() == approx(1.3)
    assert model.bias.item() == approx(0.8)


def test_create_supervised_trainer_traced_with_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    example_input = torch.randn(1, 1)
    traced_model = torch.jit.trace(model, example_input)

    optimizer = SGD(traced_model.parameters(), 0.1)

    trainer = create_supervised_trainer(traced_model, optimizer, mse_loss, device='cpu')

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    assert traced_model.weight.data[0, 0].item() == approx(0.0)
    assert traced_model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert traced_model.weight.data[0, 0].item() == approx(1.3)
    assert traced_model.bias.item() == approx(0.8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    trainer = create_supervised_trainer(model, optimizer, mse_loss, device='cuda')

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert model.weight.data[0, 0].item() == approx(1.3)
    assert model.bias.item() == approx(0.8)


def test_create_supervised():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model)

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)


def test_create_supervised_on_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model, device='cpu')

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)


def test_create_supervised_evaluator_traced_on_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    example_input = torch.randn(1, 1)
    traced_model = torch.jit.trace(model, example_input)

    evaluator = create_supervised_evaluator(traced_model, device='cpu')

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert traced_model.weight.data[0, 0].item() == approx(0.0)
    assert traced_model.bias.item() == approx(0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_on_cuda():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model, device='cuda')

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)


def test_create_supervised_with_metrics():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model, metrics={'mse': MeanSquaredError()})

    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [4.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    assert state.metrics['mse'] == 12.5


def test_reset_should_terminate():

    def update_fn(engine, batch):
        pass

    engine = Engine(update_fn)

    @engine.on(Events.ITERATION_COMPLETED)
    def terminate_on_iteration_10(engine):
        if engine.state.iteration == 10:
            engine.terminate()

    engine.run([0] * 20)
    assert engine.state.iteration == 10

    engine.run([0] * 20)
    assert engine.state.iteration == 10


def test_state_repr():

    data = [0, 1, 2, 3, 4, 5]
    max_epochs = 1
    metrics = {"accuracy": Mock()}
    state = State(dataloader=data, max_epochs=max_epochs, metrics=metrics)
    s = repr(state)
    assert "iteration: 0" in s
    assert "epoch: 0" in s
    assert "max_epochs: 1" in s
    assert "dataloader" in s
    assert "metrics" in s
    assert "output" in s
    assert "batch" in s


def test_alter_batch():

    small_shape = (1, 2, 2)
    large_shape = (1, 3, 3)

    small_loader = torch.randint(0, 256, size=(30, ) + small_shape)
    large_loader = torch.randint(0, 256, size=(20, ) + large_shape)

    switch_iteration = 50

    def should_take_large_img(i):
        return i >= switch_iteration

    def update_fn(engine, batch):
        i = engine.state.iteration
        if i < switch_iteration:
            assert batch.shape == small_shape
            assert (small_loader[(i - 1) % len(small_loader), ...] == batch).all()
        else:
            assert batch.shape == large_shape
            assert (large_loader[(i - switch_iteration) % len(large_loader), ...] == batch).all()

    trainer = Engine(update_fn)

    def cycle(seq):
        while True:
            for i in seq:
                yield i

    small_loader_iter = cycle(small_loader)
    large_loader_iter = cycle(large_loader)

    @trainer.on(Events.ITERATION_STARTED)
    def choose_batch(engine):
        i = engine.state.iteration
        if should_take_large_img(i):
            batch = next(large_loader_iter)
        else:
            batch = next(small_loader_iter)

        engine.state.batch = batch

    num_epochs = 5
    num_iters = 25
    data = list(range(num_iters))
    trainer.run(data, num_epochs)
