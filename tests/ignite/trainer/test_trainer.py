from __future__ import division

import uuid
import numpy as np
from mock import call, MagicMock, Mock
from pytest import raises

from ignite.trainer import Trainer, TrainingEvents



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


def test_adding_handler_for_non_existent_event_throws_error():
    trainer = Trainer(MagicMock(), MagicMock(), MagicMock(), MagicMock())

    event_name = uuid.uuid4()
    while event_name in TrainingEvents.__members__.values():
        event_name = uuid.uuid4()

    with raises(ValueError):
        trainer.add_event_handler(event_name, lambda x: x)


def test_exception_handler_called_on_error():
    training_update_function = MagicMock(side_effect=ValueError())

    trainer = Trainer([1], training_update_function, MagicMock(), MagicMock())
    exception_handler = MagicMock()
    trainer.add_event_handler(TrainingEvents.EXCEPTION_RAISED, exception_handler)

    with raises(ValueError):
        trainer.run()

    exception_handler.assert_called_once_with(trainer)


def test_adding_multiple_event_handlers():
    trainer = Trainer([1], MagicMock(return_value=1), MagicMock(), MagicMock())
    handlers = [MagicMock(), MagicMock()]
    for handler in handlers:
        trainer.add_event_handler(TrainingEvents.TRAINING_STARTED, handler)

    trainer.run(validate_every_epoch=False)
    for handler in handlers:
        handler.assert_called_once_with(trainer)


def test_args_and_kwargs_are_passed_to_event():
    trainer = Trainer([1], MagicMock(return_value=1), MagicMock(), MagicMock())
    kwargs = {'a': 'a', 'b': 'b'}
    args = (1, 2, 3)
    handlers = []
    for event in TrainingEvents:
        handler = MagicMock()
        trainer.add_event_handler(event, handler, *args, **kwargs)
        handlers.append(handler)

    trainer.run(max_epochs=1, validate_every_epoch=False)
    called_handlers = [handle for handle in handlers if handle.called]
    assert len(called_handlers) > 0

    for handler in called_handlers:
        handler_args, handler_kwargs = handler.call_args
        assert handler_args[0] == trainer
        assert handler_args[1::] == args
        assert handler_kwargs == kwargs


def test_current_epoch_counter_increases_every_epoch():
    trainer = Trainer([1], MagicMock(return_value=1), MagicMock(), MagicMock())
    max_epochs = 5

    class EpochCounter(object):
        def __init__(self):
            self.current_epoch_count = 0

        def __call__(self, trainer):
            assert trainer.current_epoch == self.current_epoch_count
            self.current_epoch_count += 1

    trainer.add_event_handler(TrainingEvents.EPOCH_STARTED, EpochCounter())

    trainer.run(max_epochs=max_epochs, validate_every_epoch=False)

    assert trainer.current_epoch == max_epochs


def test_current_iteration_counter_increases_every_iteration():
    training_batches = [1, 2, 3]
    trainer = Trainer(training_batches, MagicMock(return_value=1), MagicMock(), MagicMock())
    max_epochs = 5

    class IterationCounter(object):
        def __init__(self):
            self.current_iteration_count = 0

        def __call__(self, trainer):
            assert trainer.current_iteration == self.current_iteration_count
            self.current_iteration_count += 1

    trainer.add_event_handler(TrainingEvents.TRAINING_ITERATION_STARTED, IterationCounter())

    trainer.run(max_epochs=max_epochs, validate_every_epoch=False)

    assert trainer.current_iteration == max_epochs * len(training_batches)


def test_current_validation_iteration_counter_increases_every_iteration():
    validation_batches = [1, 2, 3]
    trainer = Trainer([1], MagicMock(return_value=1), validation_batches, MagicMock(return_value=1))
    max_epochs = 5

    class IterationCounter(object):
        def __init__(self):
            self.current_iteration_count = 0
            self.total_count = 0

        def __call__(self, trainer):
            assert trainer.current_validation_iteration == self.current_iteration_count
            self.current_iteration_count += 1
            self.total_count += 1

        def clear(self):
            self.current_iteration_count = 0

    iteration_counter = IterationCounter()

    def clear_counter(trainer, counter):
        counter.clear()

    trainer.add_event_handler(TrainingEvents.VALIDATION_STARTING, clear_counter, iteration_counter)
    trainer.add_event_handler(TrainingEvents.VALIDATION_ITERATION_STARTED, iteration_counter)

    trainer.run(max_epochs=max_epochs, validate_every_epoch=True)

    assert iteration_counter.total_count == max_epochs * len(validation_batches)


def test_validate_is_called_every_epoch_by_default():
    trainer = Trainer([1], MagicMock(return_value=1), [1], MagicMock())
    trainer.validate = MagicMock()

    max_epochs = 5
    trainer.run(max_epochs=max_epochs)
    assert trainer.validate.call_count == max_epochs


def test_validate_not_called_if_validate_every_epoch_is_false():
    trainer = Trainer([1], MagicMock(return_value=1), MagicMock(), MagicMock())
    trainer.validate = MagicMock()

    max_epochs = 5
    trainer.run(max_epochs=max_epochs, validate_every_epoch=False)
    assert trainer.validate.call_count == 0


def test_stopping_criterion_is_max_epochs():
    trainer = Trainer([1], MagicMock(return_value=1), MagicMock(), MagicMock())
    max_epochs = 5
    trainer.run(max_epochs=max_epochs, validate_every_epoch=False)
    assert trainer.current_epoch == max_epochs


def test_terminate_at_end_of_epoch_stops_training():
    max_epochs = 5
    last_epoch_to_run = 3

    def end_of_epoch_handler(trainer):
        if trainer.current_epoch == last_epoch_to_run:
            trainer.terminate()

    trainer = Trainer([1], MagicMock(return_value=1), MagicMock(), MagicMock())
    trainer.add_event_handler(TrainingEvents.EPOCH_COMPLETED, end_of_epoch_handler)

    assert not trainer.should_terminate

    trainer.run(max_epochs=max_epochs, validate_every_epoch=False)

    assert trainer.current_epoch == last_epoch_to_run + 1  # counter is incremented at end of loop
    assert trainer.should_terminate


def test_terminate_at_start_of_epoch_stops_training_after_completing_iteration():
    max_epochs = 5
    epoch_to_terminate_on = 3
    batches_per_epoch = [1, 2, 3]

    def start_of_epoch_handler(trainer):
        if trainer.current_epoch == epoch_to_terminate_on:
            trainer.terminate()

    trainer = Trainer(batches_per_epoch, MagicMock(return_value=1), MagicMock(), MagicMock())
    trainer.add_event_handler(TrainingEvents.EPOCH_STARTED, start_of_epoch_handler)

    assert not trainer.should_terminate

    trainer.run(max_epochs=max_epochs, validate_every_epoch=False)

    # epoch is not completed so counter is not incremented
    assert trainer.current_epoch == epoch_to_terminate_on
    assert trainer.should_terminate
    # completes first iteration
    assert trainer.current_iteration == (epoch_to_terminate_on * len(batches_per_epoch)) + 1


def test_terminate_stops_training_mid_epoch():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch + 3  # i.e. part way through the 2nd epoch
    trainer = Trainer(training_data=[None] * num_iterations_per_epoch,
                      training_update_function=MagicMock(return_value=1),
                      validation_data=MagicMock(),
                      validation_inference_function=MagicMock())

    def end_of_iteration_handler(trainer):
        if trainer.current_iteration == iteration_to_stop:
            trainer.terminate()

    trainer.add_event_handler(TrainingEvents.TRAINING_ITERATION_STARTED, end_of_iteration_handler)
    trainer.run(max_epochs=3, validate_every_epoch=False)
    assert (trainer.current_iteration == iteration_to_stop +
            1)  # completes the iteration when terminate called
    assert trainer.current_epoch == np.ceil(
        iteration_to_stop / num_iterations_per_epoch) - 1  # it starts from 0


def test_terminate_stops_trainer_when_called_during_validation():
    num_iterations_per_epoch = 10
    iteration_to_stop = 3  # i.e. part way through the 2nd validation run
    epoch_to_stop = 2
    trainer = Trainer(training_data=[None] * num_iterations_per_epoch,
                      training_update_function=MagicMock(return_value=1),
                      validation_data=[None] * num_iterations_per_epoch,
                      validation_inference_function=MagicMock(return_value=1))

    def end_of_iteration_handler(trainer):
        if (trainer.current_epoch == epoch_to_stop and trainer.current_validation_iteration == iteration_to_stop):

            trainer.terminate()

    trainer.add_event_handler(TrainingEvents.VALIDATION_ITERATION_STARTED, end_of_iteration_handler)
    trainer.run(max_epochs=4, validate_every_epoch=True)

    assert trainer.current_epoch == epoch_to_stop
    # should complete the iteration when terminate called
    assert trainer.current_validation_iteration == iteration_to_stop + 1
    assert trainer.current_iteration == (epoch_to_stop + 1) * num_iterations_per_epoch


def test_terminate_after_training_iteration_skips_validation_run():
    num_iterations_per_epoch = 10
    iteration_to_stop = num_iterations_per_epoch - 1
    trainer = Trainer(training_data=[None] * num_iterations_per_epoch,
                      training_update_function=MagicMock(return_value=1),
                      validation_data=MagicMock(),
                      validation_inference_function=MagicMock())

    def end_of_iteration_handler(trainer):
        if trainer.current_iteration == iteration_to_stop:
            trainer.terminate()

    trainer.validate = MagicMock()

    trainer.add_event_handler(TrainingEvents.TRAINING_ITERATION_STARTED, end_of_iteration_handler)
    trainer.run(max_epochs=3, validate_every_epoch=True)
    assert trainer.validate.call_count == 0


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

    trainer = Trainer(training_data=training_data,
                      validation_data=MagicMock(),
                      training_update_function=MagicMock(return_value=1),
                      validation_inference_function=MagicMock())

    mock_manager = Mock()
    iteration_started = Mock()
    trainer.add_event_handler(TrainingEvents.TRAINING_ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    trainer.add_event_handler(TrainingEvents.TRAINING_ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    trainer.run(max_epochs=max_epochs, validate_every_epoch=False)

    assert iteration_started.call_count == num_batches * max_epochs
    assert iteration_complete.call_count == num_batches * max_epochs

    expected_calls = []
    for i in range(max_epochs * num_batches):
        expected_calls.append(call.iteration_started(trainer))
        expected_calls.append(call.iteration_complete(trainer))

    assert mock_manager.mock_calls == expected_calls


def test_validation_iteration_events_are_fired():
    max_epochs = 5
    num_batches = 3
    validation_data = _create_mock_data_loader(max_epochs, num_batches)

    trainer = Trainer(training_data=[None],
                      validation_data=validation_data,
                      training_update_function=MagicMock(return_value=1),
                      validation_inference_function=MagicMock(return_value=1))

    mock_manager = Mock()
    iteration_started = Mock()
    trainer.add_event_handler(TrainingEvents.VALIDATION_ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    trainer.add_event_handler(TrainingEvents.VALIDATION_ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    trainer.run(max_epochs=max_epochs)

    assert iteration_started.call_count == num_batches * max_epochs
    assert iteration_complete.call_count == num_batches * max_epochs

    expected_calls = []
    for i in range(max_epochs * num_batches):
        expected_calls.append(call.iteration_started(trainer))
        expected_calls.append(call.iteration_complete(trainer))

    assert mock_manager.mock_calls == expected_calls


def test_validation_iteration_events_are_fired_when_validate_is_called_explicitly():
    max_epochs = 5
    num_batches = 3
    validation_data = _create_mock_data_loader(max_epochs, num_batches)

    trainer = Trainer(training_data=[None],
                      validation_data=validation_data,
                      training_update_function=MagicMock(),
                      validation_inference_function=MagicMock(return_value=1))

    mock_manager = Mock()
    iteration_started = Mock()
    trainer.add_event_handler(TrainingEvents.VALIDATION_ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    trainer.add_event_handler(TrainingEvents.VALIDATION_ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    assert iteration_started.call_count == 0
    assert iteration_complete.call_count == 0

    trainer.validate()

    assert iteration_started.call_count == num_batches
    assert iteration_complete.call_count == num_batches

    #TODO add test to assure history is written to from trainer