import math
import os
from unittest.mock import ANY, MagicMock, Mock, call

import pytest
import torch
import trains

import ignite.distributed as idist
from ignite.contrib.handlers.trains_logger import *
from ignite.engine import Engine, Events, State
from ignite.handlers import Checkpoint


def test_optimizer_params_handler_wrong_setup():

    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler OptimizerParamsHandler works only with TrainsLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():

    optimizer = torch.optim.SGD([torch.Tensor(0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.trains_logger.report_scalar.assert_called_once_with(iteration=123, series="0", title="lr", value=0.01)

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.trains_logger.report_scalar.assert_called_once_with(
        iteration=123, series="0", title="generator/lr", value=0.01
    )


def test_output_handler_with_wrong_logger_type():

    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler OutputHandler works only with TrainsLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform(dirname):

    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.trains_logger.report_scalar.assert_called_once_with(
        iteration=123, series="output", title="tag", value=12345
    )

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.trains_logger.report_scalar.assert_called_once_with(
        iteration=123, series="loss", title="another_tag", value=12345
    )


def test_output_handler_metric_names(dirname):

    wrapper = OutputHandler("tag", metric_names=["a", "b"])
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.trains_logger.report_scalar.call_count == 2
    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [
            call(title="tag", series="a", iteration=5, value=12.23),
            call(title="tag", series="b", iteration=5, value=23.45),
        ],
        any_order=True,
    )

    wrapper = OutputHandler("tag", metric_names=["a", "c"])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    with pytest.warns(UserWarning):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.trains_logger.report_scalar.call_count == 1
    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="a", iteration=7, value=55.56)], any_order=True
    )

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.trains_logger.report_scalar.call_count == 2
    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [
            call(title="tag", series="a", iteration=5, value=12.23),
            call(title="tag", series="b", iteration=5, value=23.45),
        ],
        any_order=True,
    )


def test_output_handler_both(dirname):

    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.trains_logger.report_scalar.call_count == 3
    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [
            call(title="tag", series="a", iteration=5, value=12.23),
            call(title="tag", series="b", iteration=5, value=23.45),
            call(title="tag", series="loss", iteration=5, value=12345),
        ],
        any_order=True,
    )


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return "a"

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

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

    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.trains_logger.report_scalar.call_count == 1
    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="loss", iteration=mock_another_engine.state.epoch, value=mock_engine.state.output)]
    )

    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.trains_logger.report_scalar.call_count == 2
    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="loss", iteration=mock_another_engine.state.epoch, value=mock_engine.state.output)]
    )


def test_output_handler_with_global_step_transform():
    def global_step_transform(*args, **kwargs):
        return 10

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.trains_logger.report_scalar.call_count == 1
    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="loss", iteration=10, value=12345)]
    )


def test_weights_scalar_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        WeightsScalarHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    with pytest.raises(TypeError, match="Argument reduction should be callable"):
        WeightsScalarHandler(model, reduction=123)

    with pytest.raises(ValueError, match="Output of the reduction function should be a scalar"):
        WeightsScalarHandler(model, reduction=lambda x: x)

    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler WeightsScalarHandler works only with TrainsLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_scalar_handler(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsScalarHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TrainsLogger)
        mock_logger.trains_logger = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.trains_logger.report_scalar.call_count == 4
        mock_logger.trains_logger.report_scalar.assert_has_calls(
            [
                call(title=tag_prefix + "weights_norm/fc1", series="weight", iteration=5, value=0.0),
                call(title=tag_prefix + "weights_norm/fc1", series="bias", iteration=5, value=0.0),
                call(title=tag_prefix + "weights_norm/fc2", series="weight", iteration=5, value=12.0),
                call(title=tag_prefix + "weights_norm/fc2", series="bias", iteration=5, value=math.sqrt(12.0)),
            ],
            any_order=True,
        )

    _test()
    _test(tag="tag")


def test_weights_scalar_handler_frozen_layers(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [
            call(title="weights_norm/fc2", series="weight", iteration=5, value=12.0),
            call(title="weights_norm/fc2", series="bias", iteration=5, value=math.sqrt(12.0)),
        ],
        any_order=True,
    )

    with pytest.raises(AssertionError):
        mock_logger.trains_logger.report_scalar.assert_has_calls(
            [
                call(title="weights_norm/fc1", series="weight", iteration=5, value=12.0),
                call(title="weights_norm/fc1", series="bias", iteration=5, value=math.sqrt(12.0)),
            ],
            any_order=True,
        )

    assert mock_logger.trains_logger.report_scalar.call_count == 2


def test_weights_hist_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        WeightsHistHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'WeightsHistHandler' works only with TrainsLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_hist_handler(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TrainsLogger)
        mock_logger.grad_helper = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.grad_helper.add_histogram.call_count == 4
        mock_logger.grad_helper.add_histogram.assert_has_calls(
            [
                call(title=tag_prefix + "weights_fc1", hist_data=ANY, series="weight", step=5),
                call(title=tag_prefix + "weights_fc1", hist_data=ANY, series="bias", step=5),
                call(title=tag_prefix + "weights_fc2", hist_data=ANY, series="weight", step=5),
                call(title=tag_prefix + "weights_fc2", hist_data=ANY, series="bias", step=5),
            ],
            any_order=True,
        )

    _test()
    _test(tag="tag")


def test_weights_hist_handler_frozen_layers(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.grad_helper = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.grad_helper.add_histogram.assert_has_calls(
        [
            call(title="weights_fc2", hist_data=ANY, series="weight", step=5),
            call(title="weights_fc2", hist_data=ANY, series="bias", step=5),
        ],
        any_order=True,
    )

    with pytest.raises(AssertionError):
        mock_logger.grad_helper.add_histogram.assert_has_calls(
            [
                call(title="weights_fc1", hist_data=ANY, series="weight", step=5),
                call(title="weights_fc1", hist_data=ANY, series="bias", step=5),
            ],
            any_order=True,
        )
    assert mock_logger.grad_helper.add_histogram.call_count == 2


def test_grads_scalar_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        GradsScalarHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    with pytest.raises(TypeError, match="Argument reduction should be callable"):
        GradsScalarHandler(model, reduction=123)

    wrapper = GradsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler GradsScalarHandler works only with TrainsLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_scalar_handler(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsScalarHandler(model, reduction=norm_mock, tag=tag)
        mock_logger = MagicMock(spec=TrainsLogger)
        mock_logger.trains_logger = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        norm_mock.reset_mock()

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        mock_logger.trains_logger.report_scalar.assert_has_calls(
            [
                call(
                    title=tag_prefix + "grads_norm/fc1", value=ANY, series="weight", iteration=mock_engine.state.epoch
                ),
                call(title=tag_prefix + "grads_norm/fc1", value=ANY, series="bias", iteration=mock_engine.state.epoch),
                call(
                    title=tag_prefix + "grads_norm/fc2", value=ANY, series="weight", iteration=mock_engine.state.epoch
                ),
                call(title=tag_prefix + "grads_norm/fc2", value=ANY, series="bias", iteration=mock_engine.state.epoch),
            ],
            any_order=True,
        )
        assert mock_logger.trains_logger.report_scalar.call_count == 4
        assert norm_mock.call_count == 4

    _test()
    _test(tag="tag")


def test_grads_scalar_handler_frozen_layers(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = GradsScalarHandler(model, reduction=norm_mock)
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.trains_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    norm_mock.reset_mock()

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.trains_logger.report_scalar.assert_has_calls(
        [
            call(title="grads_norm/fc2", value=ANY, series="weight", iteration=mock_engine.state.epoch),
            call(title="grads_norm/fc2", value=ANY, series="bias", iteration=mock_engine.state.epoch),
        ],
        any_order=True,
    )

    with pytest.raises(AssertionError):
        mock_logger.trains_logger.report_scalar.assert_has_calls(
            [call(title="grads_norm/fc1", value=ANY, iteration=5), call("grads_norm/fc1", ANY, 5)], any_order=True
        )
    assert mock_logger.trains_logger.report_scalar.call_count == 2
    assert norm_mock.call_count == 2


def test_grads_hist_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        GradsHistHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'GradsHistHandler' works only with TrainsLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_hist_handler(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TrainsLogger)
        mock_logger.grad_helper = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.grad_helper.add_histogram.call_count == 4
        mock_logger.grad_helper.add_histogram.assert_has_calls(
            [
                call(title=tag_prefix + "grads_fc1", hist_data=ANY, series="weight", step=5),
                call(title=tag_prefix + "grads_fc1", hist_data=ANY, series="bias", step=5),
                call(title=tag_prefix + "grads_fc2", hist_data=ANY, series="weight", step=5),
                call(title=tag_prefix + "grads_fc2", hist_data=ANY, series="bias", step=5),
            ],
            any_order=True,
        )

    _test()
    _test(tag="tag")


def test_grads_hist_frozen_layers(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock(spec=TrainsLogger)
    mock_logger.grad_helper = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.grad_helper.add_histogram.call_count == 2
    mock_logger.grad_helper.add_histogram.assert_has_calls(
        [
            call(title="grads_fc2", hist_data=ANY, series="weight", step=5),
            call(title="grads_fc2", hist_data=ANY, series="bias", step=5),
        ],
        any_order=True,
    )

    with pytest.raises(AssertionError):
        mock_logger.grad_helper.add_histogram.assert_has_calls(
            [
                call(title="grads_fc1", hist_data=ANY, series="weight", step=5),
                call(title="grads_fc1", hist_data=ANY, series="bias", step=5),
            ],
            any_order=True,
        )


def test_integration(dirname):

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)

    TrainsLogger.set_bypass_mode(True)
    logger = TrainsLogger(output_uri=dirname)

    def dummy_handler(engine, logger, event_name):
        global_step = engine.state.get_event_attrib_value(event_name)
        logger.trains_logger.report_scalar(title="", series="", value="test_value", iteration=global_step)

    logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

    trainer.run(data, max_epochs=n_epochs)
    logger.close()


def test_integration_as_context_manager(dirname):

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    TrainsLogger.set_bypass_mode(True)
    with TrainsLogger(output_uri=dirname) as trains_logger:

        trainer = Engine(update_fn)

        def dummy_handler(engine, logger, event_name):
            global_step = engine.state.get_event_attrib_value(event_name)
            logger.trains_logger.report_scalar(title="", series="", value="test_value", iteration=global_step)

        trains_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

        trainer.run(data, max_epochs=n_epochs)


def test_trains_disk_saver_integration():
    model = torch.nn.Module()
    to_save_serializable = {"model": model}

    mock_logger = MagicMock(spec=TrainsLogger)
    trains.Task.current_task = Mock(return_value=object())
    trains_saver = TrainsSaver(mock_logger)
    trains.binding.frameworks.WeightsFileHandler.create_output_model = MagicMock()

    checkpoint = Checkpoint(to_save=to_save_serializable, save_handler=trains_saver, n_saved=1)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)
    checkpoint(trainer)
    trainer.state.iteration = 1
    checkpoint(trainer)
    if trains_saver._atomic:
        assert trains.binding.frameworks.WeightsFileHandler.create_output_model.call_count == 2
    else:
        saved_files = list(os.listdir(trains_saver.dirname))
        assert len(saved_files) == 1
        assert saved_files[0] == "model_1.pt"


def _test_trains_disk_saver_integration_no_logger(device):
    model = torch.nn.Module().to(device)
    to_save_serializable = {"model": model}

    if idist.get_rank() == 0:
        trains.Task.current_task = Mock(return_value=object())
        trains.binding.frameworks.WeightsFileHandler.create_output_model = MagicMock()

    trains_saver = TrainsSaver()

    checkpoint = Checkpoint(to_save=to_save_serializable, save_handler=trains_saver, n_saved=1)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)
    checkpoint(trainer)
    trainer.state.iteration = 1
    checkpoint(trainer)

    if idist.get_rank() == 0:
        if trains_saver._atomic:
            assert trains.binding.frameworks.WeightsFileHandler.create_output_model.call_count == 2
        else:
            saved_files = list(os.listdir(trains_saver.dirname))
            assert len(saved_files) == 1
            assert saved_files[0] == "model_1.pt"


def test_trains_disk_saver_integration_no_logger():
    _test_trains_disk_saver_integration_no_logger("cpu")


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    _test_trains_disk_saver_integration_no_logger("cpu")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_trains_disk_saver_integration_no_logger(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla(dirname):
    assert "xla" in idist.device().type
    _test_trains_disk_saver_integration_no_logger(idist.device())


def _test_trains_disk_saver_integration_no_logger_xla_nprocs(index):
    device = idist.device()
    _test_trains_disk_saver_integration_no_logger(device)

    import time

    # hack to have all proc properly sync:
    time.sleep(1)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_trains_disk_saver_integration_no_logger_xla_nprocs, args=(), nprocs=n)
