from __future__ import division
from enum import Enum

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
