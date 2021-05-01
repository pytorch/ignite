import pytest

from ignite.engine.engine import Engine, Events
from ignite.handlers import EpochOutputStore


@pytest.fixture
def dummy_evaluator():
    def dummy_process_function(engine, batch):
        return 1, 0

    dummy_evaluator = Engine(dummy_process_function)

    return dummy_evaluator


@pytest.fixture
def eos():
    return EpochOutputStore()


def test_no_transform(dummy_evaluator, eos):
    eos.attach(dummy_evaluator)
    dummy_evaluator.run(range(1))
    assert eos.data == [(1, 0)]


def test_transform(dummy_evaluator):
    eos = EpochOutputStore(output_transform=lambda x: x[0])
    eos.attach(dummy_evaluator)

    dummy_evaluator.run(range(1))
    assert eos.data == [1]


def test_reset(dummy_evaluator, eos):
    eos.attach(dummy_evaluator)
    dummy_evaluator.run(range(2))
    eos.reset()
    assert eos.data == []


def test_update_one_iteration(dummy_evaluator, eos):
    eos.attach(dummy_evaluator)
    dummy_evaluator.run(range(1))
    assert len(eos.data) == 1


def test_update_five_iterations(dummy_evaluator, eos):
    eos.attach(dummy_evaluator)
    dummy_evaluator.run(range(5))
    assert len(eos.data) == 5


def test_attatch(dummy_evaluator, eos):
    eos.attach(dummy_evaluator)
    assert dummy_evaluator.has_event_handler(eos.reset, Events.EPOCH_STARTED)
    assert dummy_evaluator.has_event_handler(eos.update, Events.ITERATION_COMPLETED)


def test_store_data(dummy_evaluator, eos):
    eos.attach(dummy_evaluator, name="eval_data")
    dummy_evaluator.run(range(1))
    assert dummy_evaluator.state.eval_data == eos.data
