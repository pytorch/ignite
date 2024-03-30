import logging
import re
from unittest.mock import MagicMock

import pytest

from ignite.engine import Engine, Events
from ignite.handlers.fbresearch_logger import FBResearchLogger  # Adjust the import path as necessary


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
