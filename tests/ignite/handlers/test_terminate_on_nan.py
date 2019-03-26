
import numpy as np
import torch

from ignite.engine import Engine, Events, State
from ignite.handlers import TerminateOnNan


def test_terminate_on_nan_and_inf():

    torch.manual_seed(12)

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)
    trainer.state = State()
    h = TerminateOnNan()

    trainer.state.output = 1.0
    h(trainer)
    assert not trainer.should_terminate

    trainer.state.output = torch.tensor(123.45)
    h(trainer)
    assert not trainer.should_terminate

    trainer.state.output = torch.asin(torch.randn(10,))
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = np.array([1.0, 2.0])
    h._output_transform = lambda x: x.tolist()
    h(trainer)
    assert not trainer.should_terminate
    h._output_transform = lambda x: x

    trainer.state.output = torch.asin(torch.randn(4, 4))
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = (10.0, 1.0 / torch.randint(0, 2, size=(4,)).type(torch.float), 1.0)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = (1.0, torch.tensor(1.0), "abc")
    h(trainer)
    assert not trainer.should_terminate

    trainer.state.output = 1.0 / torch.randint(0, 2, size=(4, 4)).type(torch.float)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = (float('nan'), 10.0)
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = float('inf')
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False

    trainer.state.output = [float('nan'), 10.0]
    h(trainer)
    assert trainer.should_terminate
    trainer.should_terminate = False


def test_with_terminate_on_nan():

    torch.manual_seed(12)

    data = [1.0, 0.8,
            (torch.rand(4, 4), torch.rand(4, 4)),
            torch.rand(5), torch.asin(torch.randn(4, 4)), 0.0, 1.0]

    def update_fn(engine, batch):
        return batch

    trainer = Engine(update_fn)
    h = TerminateOnNan()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, h)

    trainer.run(data, max_epochs=2)
    assert trainer.state.iteration == 5


def test_with_terminate_on_inf():

    torch.manual_seed(12)

    data = [1.0, 0.8, torch.rand(4, 4),
            (1.0 / torch.randint(0, 2, size=(4,)).type(torch.float), torch.tensor(1.234)),
            torch.rand(5), torch.asin(torch.randn(4, 4)), 0.0, 1.0]

    def update_fn(engine, batch):
        return batch

    trainer = Engine(update_fn)
    h = TerminateOnNan()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, h)

    trainer.run(data, max_epochs=2)
    assert trainer.state.iteration == 4


def test_without_terminate_on_nan_inf():

    data = [1.0, 0.8, torch.rand(4, 4), (torch.rand(5), torch.rand(5, 4)), 0.0, 1.0]

    def update_fn(engine, batch):
        return batch

    trainer = Engine(update_fn)
    h = TerminateOnNan()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, h)

    trainer.run(data, max_epochs=2)
    assert trainer.state.iteration == len(data) * 2
