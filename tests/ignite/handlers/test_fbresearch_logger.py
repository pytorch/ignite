import logging
import re
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from ignite.engine import create_supervised_trainer, Engine, Events
from ignite.handlers.fbresearch_logger import FBResearchLogger
from ignite.utils import setup_logger


@pytest.fixture
def mock_engine():
    engine = Engine(lambda e, b: None)
    engine.state.epoch = 1
    engine.state.max_epochs = 10
    engine.state.epoch_length = 100
    engine.state.iteration = 50
    return engine


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def fb_research_logger(mock_logger):
    yield FBResearchLogger(logger=mock_logger, show_output=True)


def test_fbresearch_logger_initialization(mock_logger):
    logger = FBResearchLogger(logger=mock_logger, show_output=True)
    assert logger.logger == mock_logger
    assert logger.show_output is True


def test_fbresearch_logger_attach(mock_engine, mock_logger):
    logger = FBResearchLogger(logger=mock_logger, show_output=True)
    logger.attach(mock_engine, name="Test", every=1)
    assert mock_engine.has_event_handler(logger.log_every, Events.ITERATION_COMPLETED)


@pytest.mark.parametrize(
    "output,expected_pattern",
    [
        ({"loss": 0.456, "accuracy": 0.789}, r"loss. *0.456.*accuracy. *0.789"),
        ((0.456, 0.789), r"0.456.*0.789"),
        ([0.456, 0.789], r"0.456.*0.789"),
    ],
)
def test_output_formatting(mock_engine, fb_research_logger, output, expected_pattern):
    # Ensure the logger correctly formats and logs the output for each type
    mock_engine.state.output = output
    fb_research_logger.attach(mock_engine, name="Test", every=1)
    mock_engine.fire_event(Events.ITERATION_COMPLETED)

    actual_output = fb_research_logger.logger.info.call_args_list[0].args[0]
    assert re.search(expected_pattern, actual_output)


def test_logger_type_support():
    model = nn.Linear(10, 5)
    opt = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    data = [(torch.rand(4, 10), torch.randint(0, 5, size=(4,))) for _ in range(100)]

    trainer = create_supervised_trainer(model, opt, criterion)

    logger = setup_logger("trainer", level=logging.INFO)
    logger = FBResearchLogger(logger=logger, show_output=True)
    logger.attach(trainer, name="Train", every=20, optimizer=opt)

    trainer.run(data, max_epochs=4)
    trainer.state.output = {"loss": 4.2}
    trainer.fire_event(Events.ITERATION_COMPLETED)
    trainer.state.output = "4.2"
    trainer.fire_event(Events.ITERATION_COMPLETED)
    trainer.state.output = [4.2, 4.2]
    trainer.fire_event(Events.ITERATION_COMPLETED)
    trainer.state.output = (4.2, 4.2)
    trainer.fire_event(Events.ITERATION_COMPLETED)


def test_fbrlogger_with_output_transform(mock_logger):
    trainer = Engine(lambda e, b: 42)
    fbr = FBResearchLogger(logger=mock_logger, show_output=True)
    fbr.attach(trainer, "Training", output_transform=lambda x: {"loss": x})
    trainer.run(data=[10], epoch_length=1, max_epochs=1)
    assert "loss: 42.0000" in fbr.logger.info.call_args_list[-2].args[0]


def test_fbrlogger_with_state_attrs(mock_logger):
    trainer = Engine(lambda e, b: 42)
    fbr = FBResearchLogger(logger=mock_logger, show_output=True)
    fbr.attach(trainer, "Training", state_attributes=["alpha", "beta", "gamma"])
    trainer.state.alpha = 3.899
    trainer.state.beta = torch.tensor(12.21)
    trainer.state.gamma = torch.tensor([21.0, 6.0])
    trainer.run(data=[10], epoch_length=1, max_epochs=1)
    attrs = "alpha: 3.8990 beta: 12.2100 gamma: [21.0000, 6.0000]"
    assert attrs in fbr.logger.info.call_args_list[-2].args[0]


def test_fbrlogger_iters_values_bug(mock_logger):
    max_epochs = 15
    every = 10
    data_size = 20
    trainer = Engine(lambda e, b: 42)
    fbr = FBResearchLogger(logger=mock_logger, show_output=True)
    fbr.attach(trainer, "Training", every=every)
    trainer.run(data=range(data_size), max_epochs=max_epochs)

    expected_epoch = 1
    expected_iters = [i for i in range(every, data_size + 1, every)]
    n_calls_per_epoch = data_size // every
    i = 0
    for call_args in fbr.logger.info.call_args_list:
        msg = call_args.args[0]
        if msg.startswith("Epoch"):
            expected_iter = expected_iters[i]
            assert f"Epoch [{expected_epoch}/{max_epochs}]  [{expected_iter}/{data_size}]" in msg
            if i == n_calls_per_epoch - 1:
                expected_epoch += 1
            i += 1
            if i == n_calls_per_epoch:
                i = 0
