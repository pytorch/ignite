from __future__ import division

import uuid
import torch
from torch.nn import Linear
from torch.optim import SGD
from torch.nn.functional import mse_loss
import numpy as np
from mock import call, MagicMock, Mock
from pytest import raises, approx

from ignite.engine import Events, State
from ignite.trainer import Trainer, create_supervised_trainer


class _PicklableMagicMock(object):
    def __init__(self):
        self.uuid = str(uuid.uuid4())
        self.mock = MagicMock()

    def __getstate__(self):
        return {key: self.__dict__[key] for key in self.__dict__ if key != "mock"}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, item):
        return _PicklableMagicMock()

    def __call__(self, *args, **kwargs):
        return _PicklableMagicMock()

    def __iter__(self):
        return iter([_PicklableMagicMock()])

    def __eq__(self, other):
        return other.uuid == self.uuid

    def __repr__(self):
        return self.uuid


def test_returns_state():
    trainer = Trainer(MagicMock(return_value=1))
    state = trainer.run([])

    assert isinstance(state, State)


def test_state_attributes():
    dataloader = [1, 2, 3]
    trainer = Trainer(MagicMock(return_value=1))
    state = trainer.run(dataloader, max_epochs=3)

    assert state.iteration == 9
    assert state.output == 1
    assert state.batch == 3
    assert state.dataloader == dataloader
    assert state.epoch == 3
    assert state.max_epochs == 3


def test_default_exception_handler():
    training_update_function = MagicMock(side_effect=ValueError())
    trainer = Trainer(training_update_function)

    with raises(ValueError):
        trainer.run([1])


def test_custom_exception_handler():
    value_error = ValueError()
    training_update_function = MagicMock(side_effect=value_error)

    trainer = Trainer(training_update_function)
    exception_handler = MagicMock()
    trainer.add_event_handler(Events.EXCEPTION_RAISED, exception_handler)
    state = trainer.run([1])

    # only one call from _run_once_over_data, since the exception is swallowed
    exception_handler.assert_has_calls([call(state, value_error)])


def test_current_epoch_counter_increases_every_epoch():
    trainer = Trainer(MagicMock(return_value=1))
    max_epochs = 5

    class EpochCounter(object):
        def __init__(self):
            self.current_epoch_count = 1

        def __call__(self, state):
            assert state.epoch == self.current_epoch_count
            self.current_epoch_count += 1

    trainer.add_event_handler(Events.EPOCH_STARTED, EpochCounter())

    state = trainer.run([1], max_epochs=max_epochs)

    assert state.epoch == max_epochs


def test_current_iteration_counter_increases_every_iteration():
    training_batches = [1, 2, 3]
    trainer = Trainer(MagicMock(return_value=1))
    max_epochs = 5

    class IterationCounter(object):
        def __init__(self):
            self.current_iteration_count = 1

        def __call__(self, state):
            assert state.iteration == self.current_iteration_count
            self.current_iteration_count += 1

    trainer.add_event_handler(Events.ITERATION_STARTED, IterationCounter())

    state = trainer.run(training_batches, max_epochs=max_epochs)

    assert state.iteration == max_epochs * len(training_batches)


def test_stopping_criterion_is_max_epochs():
    trainer = Trainer(MagicMock(return_value=1))
    max_epochs = 5
    state = trainer.run([1], max_epochs=max_epochs)
    assert state.epoch == max_epochs


def test_terminate_at_end_of_epoch_stops_training():
    max_epochs = 5
    last_epoch_to_run = 3

    trainer = Trainer(MagicMock(return_value=1))

    def end_of_epoch_handler(state):
        if state.epoch == last_epoch_to_run:
            trainer.terminate()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, end_of_epoch_handler)

    assert not trainer.should_terminate

    state = trainer.run([1], max_epochs=max_epochs)

    assert state.epoch == last_epoch_to_run
    assert trainer.should_terminate


def test_terminate_at_start_of_epoch_stops_training_after_completing_iteration():
    max_epochs = 5
    epoch_to_terminate_on = 3
    batches_per_epoch = [1, 2, 3]

    trainer = Trainer(MagicMock(return_value=1))

    def start_of_epoch_handler(state):
        if state.epoch == epoch_to_terminate_on:
            trainer.terminate()

    trainer.add_event_handler(Events.EPOCH_STARTED, start_of_epoch_handler)

    assert not trainer.should_terminate

    state = trainer.run(batches_per_epoch, max_epochs=max_epochs)

    # epoch is not completed so counter is not incremented
    assert state.epoch == epoch_to_terminate_on
    assert trainer.should_terminate
    # completes first iteration
    assert state.iteration == ((epoch_to_terminate_on - 1) * len(batches_per_epoch)) + 1


def test_terminate_stops_training_mid_epoch():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch + 3  # i.e. part way through the 3rd epoch
    trainer = Trainer(MagicMock(return_value=1))

    def start_of_iteration_handler(state):
        if state.iteration == iteration_to_stop:
            trainer.terminate()

    trainer.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    state = trainer.run(data=[None] * num_iterations_per_epoch, max_epochs=3)
    # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
    assert (state.iteration == iteration_to_stop)
    assert state.epoch == np.ceil(iteration_to_stop / num_iterations_per_epoch)  # it starts from 0


def _create_mock_data_loader(epochs, batches_per_epoch):
    batches = [MagicMock()] * batches_per_epoch
    data_loader_manager = MagicMock()
    batch_iterators = [iter(batches) for _ in range(epochs)]

    data_loader_manager.__iter__.side_effect = batch_iterators

    return data_loader_manager


def test_training_iteration_events_are_fired():
    max_epochs = 5
    num_batches = 3
    data = _create_mock_data_loader(max_epochs, num_batches)

    trainer = Trainer(MagicMock(return_value=1))

    mock_manager = Mock()
    iteration_started = Mock()
    trainer.add_event_handler(Events.ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    state = trainer.run(data, max_epochs=max_epochs)

    assert iteration_started.call_count == num_batches * max_epochs
    assert iteration_complete.call_count == num_batches * max_epochs

    expected_calls = []
    for i in range(max_epochs * num_batches):
        expected_calls.append(call.iteration_started(state))
        expected_calls.append(call.iteration_complete(state))

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

    assert model.weight.data[0, 0] == approx(0.0)
    assert model.bias.data[0] == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert model.weight.data[0, 0] == approx(1.3)
    assert model.bias.data[0] == approx(0.8)
