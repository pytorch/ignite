import math
import sys
import warnings
from unittest.mock import ANY, call, MagicMock

import pytest
import torch

import ignite.distributed as idist
from ignite.contrib.handlers.neptune_logger import (
    global_step_from_engine,
    GradsScalarHandler,
    NeptuneLogger,
    NeptuneSaver,
    OptimizerParamsHandler,
    OutputHandler,
    WeightsScalarHandler,
)
from ignite.engine import Engine, Events, State
from ignite.handlers.checkpoint import Checkpoint


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
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metric.assert_called_once_with("lr/group_0", y=0.01, x=123)

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metric.assert_called_once_with("generator/lr/group_0", y=0.01, x=123)


def test_output_handler_with_wrong_logger_type():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(TypeError, match="Handler OutputHandler works only with NeptuneLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metric.assert_called_once_with("tag/output", y=12345, x=123)

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.log_metric.assert_called_once_with("another_tag/loss", y=12345, x=123)


def test_output_handler_metric_names():
    wrapper = OutputHandler("tag", metric_names=["a", "b"])
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metric.call_count == 2
    mock_logger.log_metric.assert_has_calls([call("tag/a", y=12.23, x=5), call("tag/b", y=23.45, x=5)], any_order=True)

    wrapper = OutputHandler("tag", metric_names=["a"])

    mock_engine = MagicMock()
    mock_logger.log_metric = MagicMock()
    mock_engine.state = State(metrics={"a": torch.tensor([0.0, 1.0, 2.0, 3.0])})
    mock_engine.state.iteration = 5

    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metric.call_count == 4
    mock_logger.log_metric.assert_has_calls(
        [
            call("tag/a/0", y=0.0, x=5),
            call("tag/a/1", y=1.0, x=5),
            call("tag/a/2", y=2.0, x=5),
            call("tag/a/3", y=3.0, x=5),
        ],
        any_order=True,
    )

    wrapper = OutputHandler("tag", metric_names=["a", "c"])

    mock_engine = MagicMock()
    mock_logger.log_metric = MagicMock()
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()

    with pytest.warns(UserWarning):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metric.call_count == 1
    mock_logger.log_metric.assert_has_calls([call("tag/a", y=55.56, x=7)], any_order=True)

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metric.call_count == 2
    mock_logger.log_metric.assert_has_calls([call("tag/a", y=12.23, x=5), call("tag/b", y=23.45, x=5)], any_order=True)

    # log a torch tensor (ndimension = 0)
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.tensor(12.23), "b": torch.tensor(23.45)})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metric.call_count == 2
    mock_logger.log_metric.assert_has_calls(
        [call("tag/a", y=torch.tensor(12.23).item(), x=5), call("tag/b", y=torch.tensor(23.45).item(), x=5)],
        any_order=True,
    )


def test_output_handler_both():
    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.log_metric.call_count == 3
    mock_logger.log_metric.assert_has_calls(
        [call("tag/a", y=12.23, x=5), call("tag/b", y=23.45, x=5), call("tag/loss", y=12345, x=5)], any_order=True
    )


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return "a"

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    with pytest.raises(TypeError, match="global_step must be int"):
        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)


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

    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log_metric.call_count == 1
    mock_logger.log_metric.assert_has_calls(
        [call("tag/loss", y=mock_engine.state.output, x=mock_another_engine.state.epoch)]
    )

    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log_metric.call_count == 2
    mock_logger.log_metric.assert_has_calls(
        [call("tag/loss", y=mock_engine.state.output, x=mock_another_engine.state.epoch)]
    )


def test_output_handler_with_global_step_transform():
    def global_step_transform(*args, **kwargs):
        return 10

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.log_metric.call_count == 1
    mock_logger.log_metric.assert_has_calls([call("tag/loss", y=12345, x=10)])


def test_output_handler_state_attrs():
    wrapper = OutputHandler("tag", state_attributes=["alpha", "beta", "gamma"])
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 5
    mock_engine.state.alpha = 3.899
    mock_engine.state.beta = torch.tensor(12.23)
    mock_engine.state.gamma = torch.tensor([21.0, 6.0])

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.log_metric.call_count == 4
    mock_logger.log_metric.assert_has_calls(
        [
            call("tag/alpha", y=3.899, x=5),
            call("tag/beta", y=torch.tensor(12.23).item(), x=5),
            call("tag/gamma/0", y=21.0, x=5),
            call("tag/gamma/1", y=6.0, x=5),
        ],
        any_order=True,
    )


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
        mock_logger = MagicMock(spec=NeptuneLogger)
        mock_logger.log_metric = MagicMock()
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

        assert mock_logger.log_metric.call_count == 4
        mock_logger.log_metric.assert_has_calls(
            [
                call(tag_prefix + "weights_norm/fc1/weight", y=0.0, x=5),
                call(tag_prefix + "weights_norm/fc1/bias", y=0.0, x=5),
                call(tag_prefix + "weights_norm/fc2/weight", y=12.0, x=5),
                call(tag_prefix + "weights_norm/fc2/bias", y=math.sqrt(12.0), x=5),
            ],
            any_order=True,
        )

    _test()
    _test(tag="tag")


def test_weights_scalar_handler_frozen_layers(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.log_metric.assert_has_calls(
        [call("weights_norm/fc2/weight", y=12.0, x=5), call("weights_norm/fc2/bias", y=math.sqrt(12.0), x=5)],
        any_order=True,
    )

    with pytest.raises(AssertionError):
        mock_logger.log_metric.assert_has_calls(
            [call("weights_norm/fc1/weight", y=12.0, x=5), call("weights_norm/fc1/bias", y=math.sqrt(12.0), x=5)],
            any_order=True,
        )

    assert mock_logger.log_metric.call_count == 2


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
        mock_logger = MagicMock(spec=NeptuneLogger)
        mock_logger.log_metric = MagicMock()
        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        norm_mock.reset_mock()

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

        mock_logger.log_metric.assert_has_calls(
            [
                call(tag_prefix + "grads_norm/fc1/weight", y=ANY, x=5),
                call(tag_prefix + "grads_norm/fc1/bias", y=ANY, x=5),
                call(tag_prefix + "grads_norm/fc2/weight", y=ANY, x=5),
                call(tag_prefix + "grads_norm/fc2/bias", y=ANY, x=5),
            ],
            any_order=True,
        )
        assert mock_logger.log_metric.call_count == 4
        assert norm_mock.call_count == 4

    _test()
    _test(tag="tag")


def test_grads_scalar_handler_frozen_layers(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = GradsScalarHandler(model, reduction=norm_mock)
    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_metric = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    norm_mock.reset_mock()

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.log_metric.assert_has_calls(
        [call("grads_norm/fc2/weight", y=ANY, x=5), call("grads_norm/fc2/bias", y=ANY, x=5)], any_order=True
    )

    with pytest.raises(AssertionError):
        mock_logger.log_metric.assert_has_calls(
            [call("grads_norm/fc1/weight", y=ANY, x=5), call("grads_norm/fc1/bias", y=ANY, x=5)], any_order=True
        )
    assert mock_logger.log_metric.call_count == 2
    assert norm_mock.call_count == 2


def test_integration():
    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)

    npt_logger = NeptuneLogger(offline_mode=True)

    def dummy_handler(engine, logger, event_name):
        global_step = engine.state.get_event_attrib_value(event_name)
        logger.log_metric("test_value", global_step, global_step)

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

    with NeptuneLogger(offline_mode=True) as npt_logger:
        trainer = Engine(update_fn)

        def dummy_handler(engine, logger, event_name):
            global_step = engine.state.get_event_attrib_value(event_name)
            logger.log_metric("test_value", global_step, global_step)

        npt_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

        trainer.run(data, max_epochs=n_epochs)


def test_neptune_saver_serializable(dirname):

    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_artifact = MagicMock()
    model = torch.nn.Module()
    to_save_serializable = {"model": model}

    saver = NeptuneSaver(mock_logger)
    fname = dirname / "test.pt"
    saver(to_save_serializable, fname)

    assert mock_logger.log_artifact.call_count == 1


def _test_neptune_saver_integration(device):

    model = torch.nn.Module().to(device)
    to_save_serializable = {"model": model}

    mock_logger = None
    if idist.get_rank() == 0:
        mock_logger = MagicMock(spec=NeptuneLogger)
        mock_logger.log_artifact = MagicMock()
        mock_logger.delete_artifacts = MagicMock()

    saver = NeptuneSaver(mock_logger)

    checkpoint = Checkpoint(to_save=to_save_serializable, save_handler=saver, n_saved=1)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)
    checkpoint(trainer)
    trainer.state.iteration = 1
    checkpoint(trainer)
    if idist.get_rank() == 0:
        assert mock_logger.log_artifact.call_count == 2
        assert mock_logger.delete_artifacts.call_count == 1


def test_neptune_saver_integration():
    _test_neptune_saver_integration("cpu")


def test_neptune_saver_non_serializable():

    mock_logger = MagicMock(spec=NeptuneLogger)
    mock_logger.log_artifact = MagicMock()

    to_save_non_serializable = {"model": lambda x: x}

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

    assert mock_logger.log_artifact.call_count == 0


@pytest.fixture
def no_site_packages():

    neptune_client_modules = {}
    for k in sys.modules:
        if "neptune" in k:
            neptune_client_modules[k] = sys.modules[k]
    for k in neptune_client_modules:
        del sys.modules[k]

    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    for k in neptune_client_modules:
        sys.modules[k] = neptune_client_modules[k]


def test_no_neptune_client(no_site_packages):

    with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires neptune-client to be installed."):
        NeptuneLogger()


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_neptune_saver_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_neptune_saver_integration(device)
