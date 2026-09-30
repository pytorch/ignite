import sys
from unittest.mock import call, MagicMock

import pytest
import torch

from ignite.engine import Engine, Events, State

from ignite.handlers.mlflow_logger import global_step_from_engine, MLflowLogger, OptimizerParamsHandler, OutputHandler


def test_output_handler_with_wrong_logger_type():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(TypeError, match="Handler 'OutputHandler' works only with MLflowLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.log_metrics.assert_called_once_with({"tag output": 12345}, step=123)

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metrics.assert_called_once_with({"another_tag loss": 12345}, step=123)


def test_output_handler_metric_names():
    wrapper = OutputHandler("tag", metric_names=["a", "b", "c"])
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45, "c": torch.tensor(10.0)})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_called_once_with({"tag a": 12.23, "tag b": 23.45, "tag c": 10.0}, step=5)

    wrapper = OutputHandler("tag", metric_names=["a"])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.tensor([0.0, 1.0, 2.0, 3.0])})
    mock_engine.state.iteration = 5

    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_has_calls(
        [call({"tag a 0": 0.0, "tag a 1": 1.0, "tag a 2": 2.0, "tag a 3": 3.0}, step=5)], any_order=True
    )

    wrapper = OutputHandler("tag", metric_names=["a", "c"])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    with pytest.warns(UserWarning):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_has_calls([call({"tag a": 55.56}, step=7)], any_order=True)

    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(
        metrics={
            "a": 123,
            "b": {"c": [2.34, {"d": 1}]},
            "c": (22, [33, -5.5], {"e": 32.1}),
        }
    )
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_called_once_with(
        {
            "tag a": 123,
            "tag b c 0": 2.34,
            "tag b c 1 d": 1,
            "tag c 0": 22,
            "tag c 1 0": 33,
            "tag c 1 1": -5.5,
            "tag c 2 e": 32.1,
        },
        step=5,
    )


def test_output_handler_both():
    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_called_once_with({"tag a": 12.23, "tag b": 23.45, "tag loss": 12345}, step=5)


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return "a"

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=MLflowLogger)
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
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.log_metrics.assert_called_once_with({"tag loss": 12345}, step=10)


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

    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log_metrics.call_count == 1
    mock_logger.log_metrics.assert_has_calls(
        [call({"tag loss": mock_engine.state.output}, step=mock_another_engine.state.epoch)]
    )

    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log_metrics.call_count == 2
    mock_logger.log_metrics.assert_has_calls(
        [call({"tag loss": mock_engine.state.output}, step=mock_another_engine.state.epoch)]
    )


def test_output_handler_state_attrs():
    wrapper = OutputHandler("tag", state_attributes=["alpha", "beta", "gamma"])
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 5
    mock_engine.state.alpha = 3.899
    mock_engine.state.beta = torch.tensor(12.21)
    mock_engine.state.gamma = torch.tensor([21.0, 6.0])

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.log_metrics.assert_called_once_with(
        {"tag alpha": 3.899, "tag beta": torch.tensor(12.21).item(), "tag gamma 0": 21.0, "tag gamma 1": 6.0}, step=5
    )


def test_optimizer_params_handler_wrong_setup():
    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(TypeError, match="Handler OptimizerParamsHandler works only with MLflowLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():
    optimizer = torch.optim.SGD([torch.tensor(0.0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metrics.assert_called_once_with({"lr group_0": 0.01}, step=123)

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=MLflowLogger)
    mock_logger.log_metrics = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metrics.assert_called_once_with({"generator lr group_0": 0.01}, step=123)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Skip on Windows")
def test_integration(dirname):
    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)

    mlflow_logger = MLflowLogger(tracking_uri=str(dirname / "mlruns"))

    true_values = []

    def dummy_handler(engine, logger, event_name):
        global_step = engine.state.get_event_attrib_value(event_name)
        v = global_step * 0.1
        true_values.append(v)
        logger.log_metrics({"test_value": v}, step=global_step)

    mlflow_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

    import mlflow

    active_run = mlflow.active_run()

    trainer.run(data, max_epochs=n_epochs)
    mlflow_logger.close()

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=str(dirname / "mlruns"))
    stored_values = client.get_metric_history(active_run.info.run_id, "test_value")

    for t, s in zip(true_values, stored_values):
        assert pytest.approx(t) == s.value


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Skip on Windows")
def test_integration_as_context_manager(dirname):
    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    true_values = []

    with MLflowLogger(str(dirname / "mlruns")) as mlflow_logger:
        trainer = Engine(update_fn)

        def dummy_handler(engine, logger, event_name):
            global_step = engine.state.get_event_attrib_value(event_name)
            v = global_step * 0.1
            true_values.append(v)
            logger.log_metrics({"test_value": v}, step=global_step)

        mlflow_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

        import mlflow

        active_run = mlflow.active_run()

        trainer.run(data, max_epochs=n_epochs)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=str(dirname / "mlruns"))
    stored_values = client.get_metric_history(active_run.info.run_id, "test_value")

    for t, s in zip(true_values, stored_values):
        assert pytest.approx(t) == s.value


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Skip on Windows")
def test_mlflow_bad_metric_name_handling(dirname):
    import mlflow

    true_values = [123.0, 23.4, 333.4]
    with MLflowLogger(str(dirname / "mlruns")) as mlflow_logger:
        active_run = mlflow.active_run()

        handler = OutputHandler(tag="training", metric_names="all")
        engine = Engine(lambda e, b: None)
        engine.state = State(metrics={"metric:0 in %": 123.0, "metric 0": 1000.0})

        with pytest.warns(UserWarning, match=r"MLflowLogger output_handler encountered an invalid metric name"):
            engine.state.epoch = 1
            handler(engine, mlflow_logger, event_name=Events.EPOCH_COMPLETED)

            for _, v in enumerate(true_values):
                engine.state.epoch += 1
                engine.state.metrics["metric 0"] = v
                handler(engine, mlflow_logger, event_name=Events.EPOCH_COMPLETED)

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=str(dirname / "mlruns"))
    stored_values = client.get_metric_history(active_run.info.run_id, "training metric 0")

    for t, s in zip([1000.0] + true_values, stored_values):
        assert t == s.value


@pytest.mark.parametrize("no_site_packages", ["mlflow"], indirect=True)
def test_no_mlflow_client(no_site_packages):
    with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires mlflow to be installed."):
        MLflowLogger()
