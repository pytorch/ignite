import torch
from mock import MagicMock, Mock, call
from pytest import approx
from torch.nn import Linear

from ignite.engine import Events
from ignite.evaluator import Evaluator, create_supervised_evaluator


def test_current_validation_iteration_counter_increases_every_iteration():
    validation_batches = [1, 2, 3]
    evaluator = Evaluator(MagicMock(return_value=1))
    num_runs = 5

    class IterationCounter(object):
        def __init__(self):
            self.current_iteration_count = 1
            self.total_count = 0

        def __call__(self, state):
            assert state.iteration == self.current_iteration_count
            self.current_iteration_count += 1
            self.total_count += 1

        def clear(self):
            self.current_iteration_count = 1

    iteration_counter = IterationCounter()

    def clear_counter(evaluator, counter):
        counter.clear()

    evaluator.add_event_handler(Events.STARTED, clear_counter, iteration_counter)
    evaluator.add_event_handler(Events.ITERATION_STARTED, iteration_counter)

    for _ in range(num_runs):
        evaluator.run(validation_batches)

    assert iteration_counter.total_count == num_runs * len(validation_batches)


# This is testing the same thing as above
def test_evaluation_iteration_events_are_fired():
    evaluator = Evaluator(MagicMock(return_value=1))

    mock_manager = Mock()
    iteration_started = Mock()
    evaluator.add_event_handler(Events.ITERATION_STARTED, iteration_started)

    iteration_complete = Mock()
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, iteration_complete)

    mock_manager.attach_mock(iteration_started, 'iteration_started')
    mock_manager.attach_mock(iteration_complete, 'iteration_complete')

    batches = [(1, 2), (3, 4), (5, 6)]
    state = evaluator.run(batches)
    assert iteration_started.call_count == len(batches)
    assert iteration_complete.call_count == len(batches)

    expected_calls = []
    for i in range(len(batches)):
        expected_calls.append(call.iteration_started(state))
        expected_calls.append(call.iteration_complete(state))

    assert mock_manager.mock_calls == expected_calls


def test_terminate_stops_evaluator_when_called_during_iteration():
    num_iterations = 10
    iteration_to_stop = 3  # i.e. part way through the 3rd validation run
    evaluator = Evaluator(MagicMock(return_value=1))

    def start_of_iteration_handler(state):
        if state.iteration == iteration_to_stop:
            evaluator.terminate()

    evaluator.add_event_handler(Events.ITERATION_STARTED, start_of_iteration_handler)
    state = evaluator.run([None] * num_iterations)

    # should complete the iteration when terminate called but not increment counter
    assert state.iteration == iteration_to_stop


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

    assert y_pred[0, 0] == approx(0.0)
    assert y_pred[1, 0] == approx(0.0)
    assert y[0, 0] == approx(3.0)
    assert y[1, 0] == approx(5.0)

    assert model.weight.data[0, 0] == approx(0.0)
    assert model.bias.data[0] == approx(0.0)
