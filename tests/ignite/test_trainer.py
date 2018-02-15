from __future__ import division

import uuid
import torch
from torch.nn import Linear
from torch.optim import SGD
from torch.nn.functional import mse_loss
import numpy as np
from mock import call, MagicMock, Mock
from pytest import raises, approx

from ignite.engine import Events
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
    trainer.run([1])

    # only one call from _run_once_over_data, since the exception is swallowed
    exception_handler.assert_has_calls([call(trainer, value_error)])


def test_current_epoch_counter_increases_every_epoch():
    trainer = Trainer(MagicMock(return_value=1))
    max_epochs = 5

    class EpochCounter(object):
        def __init__(self):
            self.current_epoch_count = 1

        def __call__(self, trainer):
            assert trainer.current_epoch == self.current_epoch_count
            self.current_epoch_count += 1

    trainer.add_event_handler(Events.EPOCH_STARTED, EpochCounter())

    trainer.run([1], max_epochs=max_epochs)

    assert trainer.current_epoch == max_epochs


def test_current_iteration_counter_increases_every_iteration():
    training_batches = [1, 2, 3]
    trainer = Trainer(MagicMock(return_value=1))
    max_epochs = 5

    class IterationCounter(object):
        def __init__(self):
            self.current_iteration_count = 1

        def __call__(self, trainer):
            assert trainer.current_iteration == self.current_iteration_count
            self.current_iteration_count += 1

    trainer.add_event_handler(Events.ITERATION_STARTED, IterationCounter())

    trainer.run(training_batches, max_epochs=max_epochs)

    assert trainer.current_iteration == max_epochs * len(training_batches)


def test_stopping_criterion_is_max_epochs():
    trainer = Trainer(MagicMock(return_value=1))
    max_epochs = 5
    trainer.run([1], max_epochs=max_epochs)
    assert trainer.current_epoch == max_epochs


def test_terminate_at_end_of_epoch_stops_training():
    max_epochs = 5
    last_epoch_to_run = 3

    def end_of_epoch_handler(trainer):
        if trainer.current_epoch == last_epoch_to_run:
            trainer.terminate()

    trainer = Trainer(MagicMock(return_value=1))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, end_of_epoch_handler)

    assert not trainer.should_terminate

    trainer.run([1], max_epochs=max_epochs)

    assert trainer.current_epoch == last_epoch_to_run
    assert trainer.should_terminate


def test_terminate_at_start_of_epoch_stops_training_after_completing_iteration():
    max_epochs = 5
    epoch_to_terminate_on = 3
    batches_per_epoch = [1, 2, 3]

    def start_of_epoch_handler(trainer):
        if trainer.current_epoch == epoch_to_terminate_on:
            trainer.terminate()

    trainer = Trainer(MagicMock(return_value=1))
    trainer.add_event_handler(Events.EPOCH_STARTED, start_of_epoch_handler)

    assert not trainer.should_terminate

    trainer.run(batches_per_epoch, max_epochs=max_epochs)

    # epoch is not completed so counter is not incremented
    assert trainer.current_epoch == epoch_to_terminate_on
    assert trainer.should_terminate
    # completes first iteration
    assert trainer.current_iteration == ((epoch_to_terminate_on - 1) * len(batches_per_epoch)) + 1


def test_terminate_stops_training_mid_epoch():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch + 3  # i.e. part way through the 3rd epoch
    trainer = Trainer(MagicMock(return_value=1))

    def start_of_iteration_handler(trainer):
        if trainer.current_iteration == iteration_to_stop:
            trainer.terminate()

    trainer.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    trainer.run(training_data=[None] * num_iterations_per_epoch, max_epochs=3)
    # completes the iteration but doesn't increment counter (this happens just before a new iteration starts)
    assert (trainer.current_iteration == iteration_to_stop)
    assert trainer.current_epoch == np.ceil(iteration_to_stop / num_iterations_per_epoch)  # it starts from 0


def _create_mock_data_loader(epochs, batches_per_epoch):
    batches = [MagicMock()] * batches_per_epoch
    data_loader_manager = MagicMock()
    batch_iterators = [iter(batches) for _ in range(epochs)]

    data_loader_manager.__iter__.side_effect = batch_iterators

    return data_loader_manager


def test_training_iteration_events_are_fired():
    max_epochs = 5
    num_batches = 3
    training_data = _create_mock_data_loader(max_epochs, num_batches)

    trainer = Trainer(MagicMock(return_value=1))

    mock_manager = Mock()
    iteration_started = Mock()
    trainer.add_event_handler(Events.ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    trainer.run(training_data, max_epochs=max_epochs)

    assert iteration_started.call_count == num_batches * max_epochs
    assert iteration_complete.call_count == num_batches * max_epochs

    expected_calls = []
    for i in range(max_epochs * num_batches):
        expected_calls.append(call.iteration_started(trainer))
        expected_calls.append(call.iteration_complete(trainer))

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

    trainer.run(data)
    loss = trainer.history[0]

    assert loss == approx(17.0)
    assert model.weight.data[0, 0] == approx(1.3)
    assert model.bias.data[0] == approx(0.8)
