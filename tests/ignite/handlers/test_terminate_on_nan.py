import torch

import pytest

from ignite.engine import Engine, Events, State
from ignite.handlers import TerminateOnNan


def test_output_type_in_terminate_on_nan():

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)
    trainer.state = State()
    h = TerminateOnNan()

    with pytest.raises(TypeError):
        trainer.state.output = "abc"
        h(trainer)

    with pytest.raises(TypeError):
        trainer.state.output = (1, 2, 3)
        h(trainer)


def test_terminate_on_nan_and_inf():

    torch.manual_seed(12)
    outputs = iter([
        1.0,
        torch.tensor(0.0),
        torch.asin(torch.randn(10,)),
        2.0,
        torch.asin(torch.randn(4, 4)),
        1.0 / torch.randint(0, 2, size=(4,)),
        3.0,
        1.0 / torch.randint(0, 2, size=(4, 4)),
        float('nan'),
        float('inf'),
    ])

    def update_fn(engine, batch):
        return next(outputs)

    trainer = Engine(update_fn)
    trainer.state = State()
    h = TerminateOnNan()

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert not trainer.should_terminate

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert not trainer.should_terminate

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert not trainer.should_terminate

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert not trainer.should_terminate

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = update_fn(trainer, None)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False


def test_with_terminate_on_nan():

    data = [1.0, 0.8, torch.rand(4, 4), torch.rand(5), torch.asin(torch.randn(4, 4)), 0.0, 1.0]

    def update_fn(engine, batch):
        return batch

    trainer = Engine(update_fn)
    h = TerminateOnNan()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, h)

    trainer.run(data, max_epochs=2)
    assert trainer.state.iteration == 5


def test_with_terminate_on_inf():

    data = [1.0, 0.8, torch.rand(4, 4),
            1.0 / torch.randint(0, 2, size=(4,)), torch.rand(5), torch.asin(torch.randn(4, 4)), 0.0, 1.0]

    def update_fn(engine, batch):
        return batch

    trainer = Engine(update_fn)
    h = TerminateOnNan()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, h)

    trainer.run(data, max_epochs=2)
    assert trainer.state.iteration == 4


def test_without_terminate_on_nan_inf():

    data = [1.0, 0.8, torch.rand(4, 4), torch.rand(5), 0.0, 1.0]

    def update_fn(engine, batch):
        return batch

    trainer = Engine(update_fn)
    h = TerminateOnNan()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, h)

    trainer.run(data, max_epochs=2)
    assert trainer.state.iteration == len(data) * 2
