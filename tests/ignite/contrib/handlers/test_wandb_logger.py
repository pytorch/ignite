from unittest.mock import call, MagicMock
import pytest
import torch

from ignite.engine import Events, State
from ignite.contrib.handlers.wandb_logger import *


def test_optimizer_params_handler_wrong_setup():
    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OptimizerParamsHandler' works only with WandBLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():
    optimizer = torch.optim.SGD([torch.Tensor(0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log.assert_called_once_with({"lr/group_0": 0.01}, step=123, sync=None)

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log.assert_called_once_with({"generator/lr/group_0": 0.01}, step=123, sync=None)


def test_output_handler_with_wrong_logger_type():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OutputHandler' works only with WandBLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.log.assert_called_once_with({"tag/output": 12345}, step=123, sync=None)

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log.assert_called_once_with({"another_tag/loss": 12345}, step=123, sync=None)


def test_output_handler_output_transform_sync():
    wrapper = OutputHandler("tag", output_transform=lambda x: x, sync=False)
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.log.assert_called_once_with({"tag/output": 12345}, step=123, sync=False)

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x}, sync=True)
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log.assert_called_once_with({"another_tag/loss": 12345}, step=123, sync=True)


def test_output_handler_metric_names():
    wrapper = OutputHandler("tag", metric_names=["a", "b"])
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 1, "b": 5})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log.assert_called_once_with({"tag/a": 1, "tag/b": 5}, step=5, sync=None)

    wrapper = OutputHandler("tag", metric_names=["a", "c"])
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log.assert_called_once_with({"tag/a": 55.56, "tag/c": "Some text"}, step=7, sync=None)

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log.assert_called_once_with({"tag/a": 12.23, "tag/b": 23.45}, step=5, sync=None)


def test_output_handler_both():
    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.log.assert_called_once_with({"tag/a": 12.23, "tag/b": 23.45, "tag/loss": 12345}, step=5, sync=None)


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return "a"

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    with pytest.raises(TypeError, match="global_step must be int"):
        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)


def test_output_handler_with_global_step_transform():
    def global_step_transform(*args, **kwargs):
        return 10

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.log.assert_called_once_with({"tag/loss": 12345}, step=10, sync=None)


def test_output_handler_with_global_step_from_engine():

    mock_another_engine = MagicMock()
    mock_another_engine.state = State()
    mock_another_engine.state.epoch = 10
    mock_another_engine.state.output = 12.345

    wrapper = OutputHandler(
        "tag",
        output_transform=lambda x: {"loss": x},
        global_step_transform=global_step_from_engine(mock_another_engine),
    )

    mock_logger = MagicMock(spec=WandBLogger)
    mock_logger.log = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.log.assert_called_once_with(
        {"tag/loss": mock_engine.state.output}, step=mock_another_engine.state.epoch, sync=None
    )

    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log.call_count == 2
    mock_logger.log.assert_has_calls(
        [call({"tag/loss": mock_engine.state.output}, step=mock_another_engine.state.epoch, sync=None)]
    )


@pytest.fixture
def no_site_packages():
    import sys

    wandb_client_modules = {}
    for k in sys.modules:
        if "wandb" in k:
            wandb_client_modules[k] = sys.modules[k]
    for k in wandb_client_modules:
        del sys.modules[k]

    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    for k in wandb_client_modules:
        sys.modules[k] = wandb_client_modules[k]


def test_no_wandb_client(no_site_packages):

    with pytest.raises(RuntimeError, match=r"This contrib module requires wandb to be installed."):
        WandBLogger()
