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
            self._fire_event(Events.STARTED)
            self._fire_event(Events.COMPLETED)
        return self.state


def test_terminate():
    engine = DummyEngine()
    assert not engine.should_terminate
    engine.terminate()
    assert engine.should_terminate


def test_invalid_process_raises_with_invalid_signature():
    engine = Engine(lambda engine, batch: None)

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
    dataloader = [1, 2, 3]
    engine = Engine(MagicMock(return_value=1))
    state = engine.run(dataloader)

    assert isinstance(state, State)


def test_state_attributes():
    dataloader = [1, 2, 3]
    engine = Engine(MagicMock(return_value=1))
    state = engine.run(dataloader, max_epochs=3)

    assert state.iteration == 9
    assert state.output == 1
    assert state.batch == 3

    assert engine.dataloader == dataloader
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
    state = engine.run([1])

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
    iteration_to_stop = num_iterations_per_epoch + 3  # i.e. part way through the 3rd epoch
    engine = Engine(MagicMock(return_value=1))

    def start_of_iteration_handler(engine):
        if engine.state.iteration == iteration_to_stop:
            engine.terminate()

    engine.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    state = engine.run(data=[None] * num_iterations_per_epoch, max_epochs=3)
    # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
    assert (state.iteration == iteration_to_stop)
    assert state.epoch == np.ceil(iteration_to_stop / num_iterations_per_epoch)  # it starts from 0


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

    state = engine.run(data, max_epochs=max_epochs)

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


def test_state_dict():

    def update(engine, batch):
        return batch

    engine = Engine(update)
    state_dict = engine.state_dict()
    assert state_dict == {'iteration': 0, 'epoch': 0, 'max_epochs': 0, 'seed': None}

    engine.run([0, 1, 2, 3, 4], max_epochs=5, seed=12345)

    state_dict = engine.state_dict()
    assert state_dict == {'iteration': 25, 'epoch': 5, 'max_epochs': 5, 'seed': 12345}


def test_load_state_dict():

    def update(engine, batch):
        return batch

    engine = Engine(update)
    true_state_dict = {'iteration': 25, 'epoch': 5, 'max_epochs': 5, 'seed': 12345}
    engine.load_state_dict(true_state_dict)

    state_dict = engine.state_dict()
    assert state_dict == true_state_dict


def _check_resume_engine(data_loader, batch_size, n_epochs, fail_iteration):

    # Setup engine with a handler to fetch batches
    n_iterations = n_epochs * len(data_loader)
    true_samples = np.zeros((n_iterations, batch_size), dtype=np.float)
    true_targets = np.zeros((n_iterations, batch_size), dtype=np.int)
    true_epochs = -1 * np.ones((n_iterations,), dtype=np.int)

    def update(engine, batch):
        return batch

    engine = Engine(update)

    def get_batch(engine):
        batch_x, batch_y = engine.state.output
        iteration = engine.state.iteration - 1
        true_samples[iteration, :] = batch_x
        true_targets[iteration, :] = batch_y
        true_epochs[iteration] = engine.state.epoch

    engine.add_event_handler(Events.ITERATION_COMPLETED, get_batch)

    # 1) Run engine
    engine.run(data_loader, max_epochs=n_epochs, seed=12345)

    # Setup another engine and run until an exception:
    failed_engine = Engine(update)

    failed_engine_samples = np.zeros((n_iterations, batch_size), dtype=np.float)
    failed_engine_targets = np.zeros((n_iterations, batch_size), dtype=np.int)

    def get_batch(engine):
        batch_x, batch_y = engine.state.output
        iteration = engine.state.iteration - 1
        failed_engine_samples[iteration, :] = batch_x
        failed_engine_targets[iteration, :] = batch_y

    def make_engine_fail(engine):
        if engine.state.iteration == fail_iteration:
            raise RuntimeError("STOP")

    failed_engine.add_event_handler(Events.ITERATION_COMPLETED, get_batch)
    failed_engine.add_event_handler(Events.ITERATION_COMPLETED, make_engine_fail)

    # 2) Run engine until the fail
    try:
        failed_engine.run(data_loader, max_epochs=n_epochs, seed=12345)
    except RuntimeError:
        pass

    # Let's check that seen targets are the same:
    assert np.all(true_targets[:fail_iteration, :] == failed_engine_targets[:fail_iteration, :])

    # 3) Resume 3rd engine from fail_iteration + 1
    resumed_engine = Engine(update)

    resumed_engine_samples = np.zeros((n_iterations, batch_size), dtype=np.float)
    resumed_engine_targets = np.zeros((n_iterations, batch_size), dtype=np.int)
    resumed_epochs = -1 * np.ones((n_iterations,), dtype=np.int)

    def get_batch(engine):
        batch_x, batch_y = engine.state.output
        iteration = engine.state.iteration - 1
        resumed_engine_samples[iteration, :] = batch_x
        resumed_engine_targets[iteration, :] = batch_y
        resumed_epochs[iteration] = engine.state.epoch

    resumed_engine.add_event_handler(Events.ITERATION_COMPLETED, get_batch)

    # Manually setup the resumed state:
    resumed_engine.state = State(iteration=failed_engine.state.iteration,
                                 epoch=failed_engine.state.epoch,
                                 max_epochs=failed_engine.state.max_epochs,
                                 seed=12345)
    resumed_engine._run_with_resume(data_loader)

    # Let's check that seen targets are the same:
    assert np.all(true_targets[fail_iteration + 1:, :] == resumed_engine_targets[fail_iteration + 1:, :])
    assert np.all(true_epochs[fail_iteration + 1:] == resumed_epochs[fail_iteration + 1:])


def test_resume_engine_data_loader():

    # Idea is to
    # 1) run engine normally and fetch batches
    # 2) run again (with same seed) and raise exception at given iteration
    # 3) resume engine from given iteration and compare batches
    #
    # ! No check of model/optimizer !
    #

    from torch.utils.data import DataLoader, Dataset

    # Setup dataset & data loader
    n_samples = 30
    x = np.arange(n_samples)

    def transform(v):
        return torch.rand(1).item()

    class TestDataset(Dataset):
        def __len__(self):
            return n_samples

        def __getitem__(self, index):
            return transform(x[index]), x[index]

    dataset = TestDataset()
    batch_size = 4
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    _check_resume_engine(data_loader, batch_size, n_epochs=3, fail_iteration=11)
    _check_resume_engine(data_loader, batch_size, n_epochs=3, fail_iteration=8)  # start of the epoch
    _check_resume_engine(data_loader, batch_size, n_epochs=3, fail_iteration=14)  # end of the epoch


# def test_resume_engine_ndarray():
#
#     # Idea is to
#     # 1) run engine normally and fetch batches
#     # 2) run again (with same seed) and raise exception at given iteration
#     # 3) resume engine from given iteration and compare batches
#     #
#     # ! No check of model/optimizer !
#     #
#     batch_size = 4
#     n_samples = 32
#     batches = np.arange(2 * n_samples).reshape(-1, 2, batch_size)
#
#     _check_resume_engine(batches, batch_size, n_epochs=3, fail_iteration=11)
#     _check_resume_engine(batches, batch_size, n_epochs=3, fail_iteration=8)  # start of the epoch
#     _check_resume_engine(batches, batch_size, n_epochs=3, fail_iteration=14)  # end of the epoch
