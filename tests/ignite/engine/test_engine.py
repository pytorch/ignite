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
from ignite.metrics import MeanSquaredError
from ignite.engine.engine import ReproducibleBatchSampler, _update_dataloader


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

    with pytest.raises(ValueError):
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
    class Custom_Events(Enum):
        TEST_EVENT = "test_event"

    # Dummy engine
    engine = Engine(lambda engine, batch: 0)
    engine.register_events(*Custom_Events)

    # Handle is never called
    handle = MagicMock()
    engine.add_event_handler(Custom_Events.TEST_EVENT, handle)
    engine.run(range(1))
    assert not handle.called

    # Advanced engine
    def process_func(engine, batch):
        engine.fire_event(Custom_Events.TEST_EVENT)

    engine = Engine(process_func)
    engine.register_events(*Custom_Events)

    # Handle should be called
    handle = MagicMock()
    engine.add_event_handler(Custom_Events.TEST_EVENT, handle)
    engine.run(range(1))
    assert handle.called


def test_custom_events_with_event_to_attr():
    class Custom_Events(Enum):
        TEST_EVENT = "test_event"

    custom_event_to_attr = {Custom_Events.TEST_EVENT: 'test_event'}

    # Dummy engine
    engine = Engine(lambda engine, batch: 0)
    engine.register_events(*Custom_Events, event_to_attr=custom_event_to_attr)

    # Handle is never called
    handle = MagicMock()
    engine.add_event_handler(Custom_Events.TEST_EVENT, handle)
    engine.run(range(1))
    assert hasattr(engine.state, 'test_event')
    assert engine.state.test_event == 0

    # Advanced engine
    def process_func(engine, batch):
        engine.fire_event(Custom_Events.TEST_EVENT)

    engine = Engine(process_func)
    engine.register_events(*Custom_Events, event_to_attr=custom_event_to_attr)

    def handle(engine):
        engine.state.test_event += 1

    engine.add_event_handler(Custom_Events.TEST_EVENT, handle)
    engine.run(range(25))
    assert engine.state.test_event == 25

    custom_event_to_attr = 'a'
    engine = Engine(lambda engine, batch: 0)
    with pytest.raises(ValueError):
        engine.register_events(*Custom_Events, event_to_attr=custom_event_to_attr)


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
    state = engine.run([0, ])

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
    iteration_to_stop = num_iterations_per_epoch + 4
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
    data_loader_manager.__len__.return_value = batches_per_epoch

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


def test_batch_values():

    def _test(data):
        # This test check the content passed to update function
        counter = [0]
        num_iters = len(data)

        def update_fn(engine, batch):
            assert batch == data[counter[0] % num_iters]
            counter[0] += 1

        engine = Engine(update_fn)
        engine.run(data, max_epochs=10)

    data = torch.randint(0, 1000, size=(256, ))
    _test(data)


class BatchChecker:

    def __init__(self, data, init_counter=0):
        self.counter = init_counter
        self.data = data
        self.true_batch = None

    def check(self, batch):
        self.true_batch = self.data[self.counter % len(self.data)]
        self.counter += 1
        return (self.true_batch == batch).all()


def test_epoch_length():

    def _test(data, max_epochs, num_iters):

        batch_checker = BatchChecker(data)

        def update_fn(engine, batch):
            assert batch_checker.check(batch), \
                "{}: {} vs {}".format(batch_checker.counter, batch_checker.true_batch, batch)

        engine = Engine(update_fn)
        engine.run(data, max_epochs=max_epochs, epoch_length=num_iters)
        if num_iters is None:
            num_iters = len(data)
        assert engine.state.iteration == num_iters * max_epochs
        assert engine.state.epoch == max_epochs

    def _test_as_iter(data, max_epochs, num_iters):

        batch_checker = BatchChecker(data)

        def update_fn(engine, batch):
            assert batch_checker.check(batch), \
                "{}: {} vs {}".format(batch_checker.counter, batch_checker.true_batch, batch)

        engine = Engine(update_fn)
        engine.run(iter(data), max_epochs=max_epochs, epoch_length=num_iters)
        if num_iters is None:
            num_iters = len(data)
        assert engine.state.iteration == num_iters * max_epochs
        assert engine.state.epoch == max_epochs

    max_epochs = 10
    num_iters = 20
    data = torch.randint(0, 1000, size=(num_iters,))
    _test(data, max_epochs, num_iters=None)
    _test(data, max_epochs, num_iters)
    _test(data, max_epochs, num_iters // 2)
    _test(data, max_epochs, num_iters * 2)

    _test_as_iter(data, 1, num_iters=None)
    _test_as_iter(data, 1, num_iters)
    _test_as_iter(data, 2, num_iters // 2)


def test_state_repr():

    data = [0, 1, 2, 3, 4, 5]
    max_epochs = 1
    metrics = {"accuracy": Mock()}
    state = State(dataloader=data, max_epochs=max_epochs, metrics=metrics)
    s = repr(state)
    assert "iteration" in s
    assert "epoch" in s
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


def test_strict_resume_from_iter():

    def _test(epoch_length=None):

        max_epochs = 5
        num_iters = 20
        data = torch.randint(0, 1000, size=(num_iters,))
        if epoch_length is None:
            epoch_length = num_iters

        for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs)):
            batch_checker = BatchChecker(data, init_counter=resume_iteration)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), \
                    "{} | {}: {} vs {}".format(
                        resume_iteration,
                        batch_checker.counter, batch_checker.true_batch, batch)

            engine = Engine(update_fn)

            @engine.on(Events.EPOCH_COMPLETED)
            def check_iteration(engine):
                assert engine.state.iteration == batch_checker.counter

            resume_state_dict = {
                "iteration": resume_iteration,
                "max_epochs": max_epochs,
                "epoch_length": epoch_length,
                "seed": 0
            }
            engine.resume(data, state_dict=resume_state_dict, strict=True)
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(30)
    _test(60)
    _test(15)


def test_strict_resume_from_epoch():

    def _test(epoch_length=None):
        max_epochs = 10
        num_iters = 20
        data = torch.randint(0, 1000, size=(num_iters,))
        if epoch_length is None:
            epoch_length = num_iters

        for resume_epoch in range(1, max_epochs):
            batch_checker = BatchChecker(data, init_counter=resume_epoch * epoch_length)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), \
                    "{} | {}: {} vs {}".format(
                        resume_epoch,
                        batch_checker.counter, batch_checker.true_batch, batch)

            engine = Engine(update_fn)
            resume_state_dict = dict(epoch=resume_epoch,
                                     max_epochs=max_epochs,
                                     epoch_length=epoch_length,
                                     seed=0)
            engine.resume(data, state_dict=resume_state_dict, strict=True)
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(30)
    _test(60)
    _test(15)


def test_resume_dataloader_from_iter():

    def _test(strict, epoch_length=None):
        max_epochs = 5
        batch_size = 4
        num_iters = 20
        data = torch.randint(0, 1000, size=(num_iters * batch_size, 3, 12, 14))
        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                                                     drop_last=False, shuffle=False)

            for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs), num_iters // 2):
                batch_checker = BatchChecker(data.reshape(-1, batch_size, 3, 12, 14),
                                             init_counter=resume_iteration if strict else 0)

                def update_fn(engine, batch):
                    assert batch_checker.check(batch), \
                        "{} {} | {}: {} vs {}".format(
                            num_workers, resume_iteration,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(iteration=resume_iteration,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=0)
                engine.resume(dataloader, state_dict=resume_state_dict, strict=strict)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs, \
                    "{}, {}, {} | {} vs {}".format(num_workers, resume_iteration, strict,
                                                   engine.state.iteration,
                                                   epoch_length * max_epochs)

    for strict in [True, False]:
        _test(strict=strict)
        _test(strict=strict, epoch_length=30)
        _test(strict=strict, epoch_length=60)
        _test(strict=strict, epoch_length=15)


def test_resume_dataloader_from_epoch():

    def _test(strict, epoch_length=None):

        max_epochs = 10
        batch_size = 4
        num_iters = 20
        data = torch.randint(0, 256, size=(num_iters * batch_size, 3, 12, 16))
        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                                                     drop_last=False, shuffle=False)

            for resume_epoch in range(1, max_epochs):
                batch_checker = BatchChecker(data.reshape(-1, batch_size, 3, 12, 16),
                                             init_counter=resume_epoch * epoch_length)

                def update_fn(engine, batch):
                    assert batch_checker.check(batch), \
                        "{} {} {} | {}: {} vs {}".format(
                            strict, num_workers, resume_epoch,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(epoch=resume_epoch,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=0)
                engine.resume(dataloader, state_dict=resume_state_dict, strict=strict)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs

    for strict in [True, False]:
        _test(strict=strict)

    _test(strict=True, epoch_length=30)
    _test(strict=True, epoch_length=60)
    _test(strict=True, epoch_length=15)


def test_reproduce_run_with_seed():

    def _test(epoch_length=None):
        max_epochs = 5
        batch_size = 4
        num_iters = 20
        data = torch.randint(0, 1000, size=(num_iters * batch_size, ))

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     drop_last=True, shuffle=True)

            ref_seen_batchs = []

            def ref_update_fn(engine, batch):
                ref_seen_batchs.append(batch)

            engine = Engine(ref_update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            seen_batchs = []

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            for i, (ref_b, b) in enumerate(zip(ref_seen_batchs, seen_batchs)):
                assert (ref_b == b).all(), "{}, {}: {} vs {}".format(num_workers, i, ref_b, b)

    _test()
    _test(30)
    _test(60)
    _test(15)


def test_resume_random_dataloader_from_epoch():

    torch.manual_seed(0)

    def _test(strict, epoch_length=None):
        max_epochs = 5
        batch_size = 4
        num_iters = 20
        data = torch.randint(0, 1000, size=(num_iters * batch_size, ))

        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory=False,
                                                     drop_last=True, shuffle=True)

            seen_batchs = []

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            for resume_epoch in range(1, max_epochs):
                batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

                def update_fn(engine, batch):
                    assert batch_checker.check(batch), \
                        "{} {} | {}: {} vs {}".format(
                            num_workers, resume_epoch,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(epoch=resume_epoch,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=12)
                engine.resume(dataloader, state_dict=resume_state_dict, strict=strict)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs

    for strict in [True, False]:
        _test(strict=strict)

    _test(True, 30)
    _test(True, 60)
    _test(True, 15)


def test_resume_random_dataloader_from_iter():

    torch.manual_seed(0)

    def _test(strict, epoch_length=None):
        max_epochs = 4
        batch_size = 5
        num_iters = 20
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))

        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     drop_last=True, shuffle=True)
            seen_batchs = []

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs), num_iters // 2):
                resume_epoch = resume_iteration // epoch_length
                batch_checker = BatchChecker(seen_batchs,
                                             init_counter=resume_iteration if strict else resume_epoch * epoch_length)

                def update_fn(engine, batch):
                    assert batch_checker.check(batch), \
                        "{} {} | {}: {} vs {}".format(
                            num_workers, resume_iteration,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(iteration=resume_iteration,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=12)
                engine.resume(dataloader, state_dict=resume_state_dict, strict=strict)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs, \
                    "{}, {}, {} | {} vs {}".format(num_workers, resume_iteration, strict,
                                                   engine.state.iteration,
                                                   epoch_length * max_epochs)

    for strict in [True, False]:
        _test(strict=strict)

    _test(True, 30)
    _test(True, 60)
    _test(True, 15)


def test_reproducible_batch_sampler():
    import torch
    from torch.utils.data import DataLoader

    data = list(range(100))
    dataloader = DataLoader(data, batch_size=12, num_workers=0, shuffle=True, drop_last=True)

    torch.manual_seed(12 + 0)
    dataloader_ = _update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))

    seen_batches = []
    num_epochs = 3
    for i in range(num_epochs):
        t = []
        for b in dataloader_:
            t.append(b)
        seen_batches.append(t)
        torch.manual_seed(12 + i + 1)

    for i in range(num_epochs - 1):
        for j in range(i + 1, num_epochs):
            assert not all([(b1 == b2).all() for b1, b2 in zip(seen_batches[i], seen_batches[j])])

    for resume_epoch in range(num_epochs):
        torch.manual_seed(12 + resume_epoch)
        dataloader_ = _update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))
        resumed_seen_batches = []
        for b in dataloader_:
            resumed_seen_batches.append(b)

        assert all([(b1 == b2).all() for b1, b2 in zip(seen_batches[resume_epoch], resumed_seen_batches)])
