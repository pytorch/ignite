import math
import warnings
from unittest.mock import MagicMock

import pytest
import torch

from ignite.engine import Engine, Events, State

from ignite.handlers.neptune_logger import (
    global_step_from_engine,
    GradsScalarHandler,
    NeptuneLogger,
    NeptuneSaver,
    OptimizerParamsHandler,
    OutputHandler,
    WeightsScalarHandler,
)


def assert_logger_called_once_with(logger, key, value):
    result = logger[key].fetch_values()
    assert len(result.value) == 1

    if isinstance(result.value[0], float):
        assert math.isclose(result.value[0], value, abs_tol=0.01)
    else:
        assert result.value[0] == value


def test_optimizer_params_handler_wrong_setup():
    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(TypeError, match="Handler OptimizerParamsHandler works only with NeptuneLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():
    optimizer = torch.optim.SGD([torch.tensor(0.0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)
    assert_logger_called_once_with(logger, "lr/group_0", 0.01)
    logger.stop()

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)
    assert_logger_called_once_with(logger, "generator/lr/group_0", 0.01)
    logger.stop()


def test_output_handler_with_wrong_logger_type():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(TypeError, match="Handler OutputHandler works only with NeptuneLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)
    assert_logger_called_once_with(logger, "tag/output", 12345)
    logger.stop()

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)
    assert_logger_called_once_with(logger, "another_tag/loss", 12345)
    logger.stop()


def test_output_handler_metric_names():
    wrapper = OutputHandler("tag", metric_names=["a", "b"])
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)

    assert_logger_called_once_with(logger, "tag/a", 12.23)
    assert_logger_called_once_with(logger, "tag/b", 23.45)
    logger.stop()

    wrapper = OutputHandler("tag", metric_names=["a"])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.tensor([0.0, 1.0, 2.0, 3.0])})
    mock_engine.state.iteration = 5

    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    wrapper(mock_engine, logger, Events.ITERATION_STARTED)

    for key, val in [("tag/a/0", 0.0), ("tag/a/1", 1.0), ("tag/a/2", 2.0), ("tag/a/3", 3.0)]:
        assert_logger_called_once_with(logger, key, val)
    logger.stop()

    wrapper = OutputHandler("tag", metric_names=["a", "c"])

    mock_engine = MagicMock()
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    with pytest.warns(UserWarning):
        wrapper(mock_engine, logger, Events.ITERATION_STARTED)

    assert_logger_called_once_with(logger, "tag/a", 55.56)
    logger.stop()

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)

    assert_logger_called_once_with(logger, "tag/a", 12.23)
    assert_logger_called_once_with(logger, "tag/b", 23.45)
    logger.stop()

    # log a torch tensor (ndimension = 0)
    wrapper = OutputHandler("tag", metric_names="all")
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.tensor(12.23), "b": torch.tensor(23.45)})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)

    assert_logger_called_once_with(logger, "tag/a", 12.23)
    assert_logger_called_once_with(logger, "tag/b", 23.45)
    logger.stop()

    wrapper = OutputHandler("tag", metric_names="all")
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State(
        metrics={
            "a": 123,
            "b": {"c": [2.34, {"d": 1}]},
            "c": (22, [33, -5.5], {"e": 32.1}),
        }
    )
    mock_engine.state.iteration = 5

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)

    assert_logger_called_once_with(logger, "tag/a", 123),
    assert_logger_called_once_with(logger, "tag/b/c/0", 2.34),
    assert_logger_called_once_with(logger, "tag/b/c/1/d", 1),
    assert_logger_called_once_with(logger, "tag/c/0", 22),
    assert_logger_called_once_with(logger, "tag/c/1/0", 33),
    assert_logger_called_once_with(logger, "tag/c/1/1", -5.5),
    assert_logger_called_once_with(logger, "tag/c/2/e", 32.1),

    logger.stop()


def test_output_handler_both():
    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, logger, Events.EPOCH_STARTED)

    assert_logger_called_once_with(logger, "tag/a", 12.23)
    assert_logger_called_once_with(logger, "tag/b", 23.45)
    assert_logger_called_once_with(logger, "tag/loss", 12345)
    logger.stop()


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return "a"

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    with pytest.raises(TypeError, match="global_step must be int"):
        wrapper(mock_engine, logger, Events.EPOCH_STARTED)

    logger.stop()


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

    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123

    wrapper(mock_engine, logger, Events.EPOCH_STARTED)
    assert_logger_called_once_with(logger, "tag/loss", mock_engine.state.output)

    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123

    wrapper(mock_engine, logger, Events.EPOCH_STARTED)

    result = logger["tag/loss"].fetch_values()
    assert len(result.value) == 2
    assert result.value[1] == mock_engine.state.output

    logger.stop()


def test_output_handler_with_global_step_transform():
    def global_step_transform(*args, **kwargs):
        return 10

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, logger, Events.EPOCH_STARTED)
    assert_logger_called_once_with(logger, "tag/loss", 12345)

    logger.stop()


def test_output_handler_state_attrs():
    wrapper = OutputHandler("tag", state_attributes=["alpha", "beta", "gamma"])
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 5
    mock_engine.state.alpha = 3.899
    mock_engine.state.beta = torch.tensor(12.23)
    mock_engine.state.gamma = torch.tensor([21.0, 6.0])

    wrapper(mock_engine, logger, Events.ITERATION_STARTED)

    assert_logger_called_once_with(logger, "tag/alpha", 3.899)
    assert_logger_called_once_with(logger, "tag/beta", 12.23)
    assert_logger_called_once_with(logger, "tag/gamma/0", 21.0)
    assert_logger_called_once_with(logger, "tag/gamma/1", 6.0)

    logger.stop()


def test_weights_scalar_handler_wrong_setup():
    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        WeightsScalarHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    with pytest.raises(TypeError, match="Argument reduction should be callable"):
        WeightsScalarHandler(model, reduction=123)

    with pytest.raises(TypeError, match="Output of the reduction function should be a scalar"):
        WeightsScalarHandler(model, reduction=lambda x: x)

    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(TypeError, match="Handler WeightsScalarHandler works only with NeptuneLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_scalar_handler(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsScalarHandler(model, tag=tag)
        logger = NeptuneLogger(
            project="tests/dry-run",
            mode="debug",
        )
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

        assert_logger_called_once_with(logger, tag_prefix + "weights_norm/fc1/weight", 0.0)
        assert_logger_called_once_with(logger, tag_prefix + "weights_norm/fc1/bias", 0.0)
        assert_logger_called_once_with(logger, tag_prefix + "weights_norm/fc2/weight", 12.0)
        assert_logger_called_once_with(logger, tag_prefix + "weights_norm/fc2/bias", math.sqrt(12.0))

        logger.stop()

    _test()
    _test(tag="tag")


def test_weights_scalar_handler_frozen_layers(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = WeightsScalarHandler(model)
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, logger, Events.EPOCH_STARTED)

    assert_logger_called_once_with(logger, "weights_norm/fc2/weight", 12.0)
    assert_logger_called_once_with(logger, "weights_norm/fc2/bias", math.sqrt(12.0))

    assert not logger.exists("weights_norm/fc1/weight")
    assert not logger.exists("weights_norm/fc1/bias")

    logger.stop()


def test_grads_scalar_handler_wrong_setup():
    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        GradsScalarHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    with pytest.raises(TypeError, match="Argument reduction should be callable"):
        GradsScalarHandler(model, reduction=123)

    wrapper = GradsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(TypeError, match="Handler GradsScalarHandler works only with NeptuneLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_scalar_handler(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsScalarHandler(model, reduction=norm_mock, tag=tag)
        logger = NeptuneLogger(
            project="tests/dry-run",
            mode="debug",
        )
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        norm_mock.reset_mock()

        wrapper(mock_engine, logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

        assert logger.exists(tag_prefix + "grads_norm/fc1/weight")
        assert logger.exists(tag_prefix + "grads_norm/fc1/bias")
        assert logger.exists(tag_prefix + "grads_norm/fc2/weight")
        assert logger.exists(tag_prefix + "grads_norm/fc2/bias")

        logger.stop()

    _test()
    _test(tag="tag")


def test_grads_scalar_handler_frozen_layers(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = GradsScalarHandler(model, reduction=norm_mock)
    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    norm_mock.reset_mock()

    wrapper(mock_engine, logger, Events.EPOCH_STARTED)

    assert logger.exists("grads_norm/fc2/weight")
    assert logger.exists("grads_norm/fc2/bias")

    assert not logger.exists("grads_norm/fc1/weight")
    assert not logger.exists("grads_norm/fc1/bias")

    logger.stop()


def test_integration():
    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)

    npt_logger = NeptuneLogger(mode="offline")

    def dummy_handler(engine, logger, event_name):
        global_step = engine.state.get_event_attrib_value(event_name)
        logger["test_value"].append(global_step, step=global_step)

    npt_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

    trainer.run(data, max_epochs=n_epochs)
    npt_logger.close()


def test_integration_as_context_manager():
    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    with NeptuneLogger(mode="offline") as npt_logger:
        trainer = Engine(update_fn)

        def dummy_handler(engine, logger, event_name):
            global_step = engine.state.get_event_attrib_value(event_name)
            logger["test_value"].append(global_step, step=global_step)

        npt_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

        trainer.run(data, max_epochs=n_epochs)


def test_neptune_saver_serializable(dirname):
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.upload = MagicMock()
    model = torch.nn.Module()
    to_save_serializable = {"model": model}

    saver = NeptuneSaver(mock_logger)
    fname = dirname / "test.pt"
    saver(to_save_serializable, fname)

    assert mock_logger[dirname / "test.pt"].upload.call_count == 1


@pytest.mark.parametrize("model, serializable", [(lambda x: x, False), (torch.nn.Module().to("cpu"), True)])
def test_neptune_saver(model, serializable):
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.upload = MagicMock()

    to_save_non_serializable = {"model": model}

    saver = NeptuneSaver(mock_logger)
    fname = "test.pt"
    try:
        with warnings.catch_warnings():
            # Ignore torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type
            # DummyModel. It won't be checked for correctness upon loading.
            warnings.simplefilter("ignore", category=UserWarning)
            saver(to_save_non_serializable, fname)
    except Exception:
        pass

    assert mock_logger["model"].upload.call_count == int(serializable)


def test_logs_version():
    from ignite import __version__
    from ignite.handlers.neptune_logger import _INTEGRATION_VERSION_KEY

    logger = NeptuneLogger(
        project="tests/dry-run",
        mode="debug",
    )
    assert logger[_INTEGRATION_VERSION_KEY].fetch() == __version__
