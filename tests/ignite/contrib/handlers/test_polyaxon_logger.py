import os
import pytest

from unittest.mock import MagicMock, call

import torch

from ignite.engine import Engine, Events, State
from ignite.contrib.handlers.polyaxon_logger import *

os.environ['POLYAXON_NO_OP'] = "1"


def test_output_handler_with_wrong_logger_type():

    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OutputHandler' works only with PolyaxonLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform():

    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.log_metrics.assert_called_once_with(step=123, **{"tag/output": 12345})

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metrics.assert_called_once_with(step=123, **{"another_tag/loss": 12345})


def test_output_handler_metric_names():

    wrapper = OutputHandler("tag", metric_names=["a", "b", "c"])
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45, "c": torch.tensor(10.0)})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_called_once_with(
        step=5,
        **{"tag/a": 12.23,
           "tag/b": 23.45,
           "tag/c": 10.0}
    )

    wrapper = OutputHandler("tag", metric_names=["a", ])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.Tensor([0.0, 1.0, 2.0, 3.0])})
    mock_engine.state.iteration = 5

    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_has_calls([
        call(step=5,
             **{"tag/a/0": 0.0,
                "tag/a/1": 1.0,
                "tag/a/2": 2.0,
                "tag/a/3": 3.0}),
    ], any_order=True)

    wrapper = OutputHandler("tag", metric_names=["a", "c"])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    with pytest.warns(UserWarning):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_has_calls([
        call(step=7, **{"tag/a": 55.56})
    ], any_order=True)

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45, "c": torch.tensor(10.0)})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_called_once_with(
        step=5,
        **{"tag/a": 12.23,
           "tag/b": 23.45,
           "tag/c": 10.0}
    )


def test_output_handler_both():

    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_called_once_with(
        step=5,
        **{"tag/a": 12.23,
           "tag/b": 23.45,
           "tag/loss": 12345}
    )


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return 'a'

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

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
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.log_metrics.assert_called_once_with(step=10, **{"tag/loss": 12345})


def test_output_handler_with_global_step_from_engine():

    mock_another_engine = MagicMock()
    mock_another_engine.state = State()
    mock_another_engine.state.epoch = 10
    mock_another_engine.state.output = 12.345

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x},
                            global_step_transform=global_step_from_engine(mock_another_engine))

    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_has_calls([call(step=mock_another_engine.state.epoch,
                                                   **{"tag/loss": mock_engine.state.output})])

    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log_metrics.call_count == 2
    mock_logger.log_metrics.assert_has_calls([call(step=mock_another_engine.state.epoch,
                                                   **{"tag/loss": mock_engine.state.output})])


def test_optimizer_params_handler_wrong_setup():

    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OptimizerParamsHandler' works only with PolyaxonLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():

    optimizer = torch.optim.SGD([torch.Tensor(0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metrics.assert_called_once_with(**{"lr/group_0": 0.01, "step": 123})

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=PolyaxonLogger)
    mock_logger.log_metrics = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metrics.assert_called_once_with(**{"generator/lr/group_0": 0.01, "step": 123})


def test_integration():

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)

    plx_logger = PolyaxonLogger()

    def dummy_handler(engine, logger, event_name):
        global_step = engine.state.get_event_attrib_value(event_name)
        logger.log_metrics(step=global_step, **{"{}".format("test_value"): global_step})

    plx_logger.attach(trainer,
                      log_handler=dummy_handler,
                      event_name=Events.EPOCH_COMPLETED)

    trainer.run(data, max_epochs=n_epochs)


def test_integration_as_context_manager():

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    with PolyaxonLogger() as plx_logger:

        trainer = Engine(update_fn)

        def dummy_handler(engine, logger, event_name):
            global_step = engine.state.get_event_attrib_value(event_name)
            logger.log_metrics(step=global_step, **{"{}".format("test_value"): global_step})

        plx_logger.attach(trainer,
                          log_handler=dummy_handler,
                          event_name=Events.EPOCH_COMPLETED)

        trainer.run(data, max_epochs=n_epochs)


@pytest.fixture
def no_site_packages():
    import sys

    polyaxon_client_modules = {}
    for k in sys.modules:
        if "polyaxon" in k:
            polyaxon_client_modules[k] = sys.modules[k]
    for k in polyaxon_client_modules:
        del sys.modules[k]

    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    for k in polyaxon_client_modules:
        sys.modules[k] = polyaxon_client_modules[k]


def test_no_polyaxon_client(no_site_packages):

    with pytest.raises(RuntimeError, match=r"This contrib module requires polyaxon-client to be installed"):
        PolyaxonLogger()
