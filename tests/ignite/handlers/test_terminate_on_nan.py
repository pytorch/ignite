import numpy as np
import pytest
import torch

from ignite.engine import Engine, Events, State
from ignite.handlers import TerminateOnNan


@pytest.mark.parametrize(
    "state_output,should_terminate",
    [
        (1.0, False),
        (torch.tensor(123.45), False),
        (torch.asin(torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])), True),
        (torch.asin(torch.randn(4, 4)), True),
        ((10.0, 1.0 / torch.tensor([1.0, 2.0, 0.0, 3.0]), 1.0), True),
        ((1.0, torch.tensor(1.0), "abc"), False),
        (1.0 / torch.randint(0, 2, size=(4, 4)).type(torch.float), True),
        ((float("nan"), 10.0), True),
        (float("inf"), True),
        ([float("nan"), 10.0], True),
        (np.array([1.0, 2.0]), False),
    ],
)
def test_terminate_on_nan_and_inf(state_output, should_terminate):
    torch.manual_seed(12)

    def update_fn(engine, batch):
        pass

    trainer = Engine(update_fn)
    trainer.state = State()
    h = TerminateOnNan()

    trainer.state.output = state_output
    if isinstance(state_output, np.ndarray):
        h._output_transform = lambda x: x.tolist()
    h(trainer)
    assert trainer.should_terminate == should_terminate


def test_with_terminate_on_nan():
    torch.manual_seed(12)

    data = [1.0, 0.8, (torch.rand(4, 4), torch.rand(4, 4)), torch.rand(5), torch.asin(torch.randn(4, 4)), 0.0, 1.0]

    def update_fn(engine, batch):
        return batch

    trainer = Engine(update_fn)
    h = TerminateOnNan()
    trainer.add_event_handler(Events.ITERATION_COMPLETED, h)

    trainer.run(data, max_epochs=2)
    assert trainer.state.iteration == 5


def test_with_terminate_on_inf():
    torch.manual_seed(12)

    data = [
        1.0,
        0.8,
        torch.rand(4, 4),
        (1.0 / torch.randint(0, 2, size=(4,)).type(torch.float), torch.tensor(1.234)),
        torch.rand(5),
        torch.asin(torch.randn(4, 4)),
        0.0,
        1.0,
    ]

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
