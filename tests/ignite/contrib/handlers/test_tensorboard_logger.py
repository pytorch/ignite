import os
import math

import pytest

from mock import MagicMock, call, ANY, Mock

import torch

from ignite.engine import Engine, Events, State
from ignite.contrib.handlers.tensorboard_logger import *


def test_optimizer_params_handler_wrong_setup():

    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OptimizerParamsHandler' works only with TensorboardLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():

    optimizer = torch.optim.SGD([torch.Tensor(0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with("lr/group_0", 0.01, 123)

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with("generator/lr/group_0", 0.01, 123)


def test_output_handler_with_wrong_logger_type():

    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OutputHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform(dirname):

    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.writer.add_scalar.assert_called_once_with("tag/output", 12345, 123)

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.writer.add_scalar.assert_called_once_with("another_tag/loss", 12345, 123)


def test_output_handler_metric_names(dirname):

    wrapper = OutputHandler("tag", metric_names=["a", "b"])
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.add_scalar.assert_has_calls([
        call("tag/a", 12.23, 5),
        call("tag/b", 23.45, 5),
    ], any_order=True)

    wrapper = OutputHandler("tag", metric_names=["a", ])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.Tensor([0.0, 1.0, 2.0, 3.0])})
    mock_engine.state.iteration = 5

    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.writer.add_scalar.call_count == 4
    mock_logger.writer.add_scalar.assert_has_calls([
        call("tag/a/0", 0.0, 5),
        call("tag/a/1", 1.0, 5),
        call("tag/a/2", 2.0, 5),
        call("tag/a/3", 3.0, 5),
    ], any_order=True)

    wrapper = OutputHandler("tag", metric_names=["a", "c"])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    with pytest.warns(UserWarning):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.writer.add_scalar.call_count == 1
    mock_logger.writer.add_scalar.assert_has_calls([
        call("tag/a", 55.56, 7),
    ], any_order=True)

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.writer.add_scalar.call_count == 2
    mock_logger.writer.add_scalar.assert_has_calls([
        call("tag/a", 12.23, 5),
        call("tag/b", 23.45, 5),
    ], any_order=True)


def test_output_handler_both(dirname):

    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.writer.add_scalar.call_count == 3
    mock_logger.writer.add_scalar.assert_has_calls([
        call("tag/a", 12.23, 5),
        call("tag/b", 23.45, 5),
        call("tag/loss", 12345, 5)
    ], any_order=True)


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return 'a'

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

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
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.writer.add_scalar.call_count == 1
    mock_logger.writer.add_scalar.assert_has_calls([call("tag/loss", 12345, 10)])


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
    with pytest.raises(RuntimeError, match="Handler 'WeightsScalarHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_scalar_handler(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsScalarHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.writer.add_scalar.call_count == 4
        mock_logger.writer.add_scalar.assert_has_calls([
            call(tag_prefix + "weights_norm/fc1/weight", 0.0, 5),
            call(tag_prefix + "weights_norm/fc1/bias", 0.0, 5),
            call(tag_prefix + "weights_norm/fc2/weight", 12.0, 5),
            call(tag_prefix + "weights_norm/fc2/bias", math.sqrt(12.0), 5),
        ], any_order=True)

    _test()
    _test(tag="tag")


def test_weights_scalar_handler_frozen_layers(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.writer.add_scalar.assert_has_calls([
        call("weights_norm/fc2/weight", 12.0, 5),
        call("weights_norm/fc2/bias", math.sqrt(12.0), 5),
    ], any_order=True)

    with pytest.raises(AssertionError):
        mock_logger.writer.add_scalar.assert_has_calls([
            call("weights_norm/fc1/weight", 12.0, 5),
            call("weights_norm/fc1/bias", math.sqrt(12.0), 5),
        ], any_order=True)

    assert mock_logger.writer.add_scalar.call_count == 2


def test_weights_hist_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        WeightsHistHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'WeightsHistHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_hist_handler(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.writer.add_histogram.call_count == 4
        mock_logger.writer.add_histogram.assert_has_calls([
            call(tag=tag_prefix + "weights/fc1/weight", values=ANY, global_step=5),
            call(tag=tag_prefix + "weights/fc1/bias", values=ANY, global_step=5),
            call(tag=tag_prefix + "weights/fc2/weight", values=ANY, global_step=5),
            call(tag=tag_prefix + "weights/fc2/bias", values=ANY, global_step=5),
        ], any_order=True)

    _test()
    _test(tag="tag")


def test_weights_hist_handler_frozen_layers(dummy_model_factory):

    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.writer.add_histogram.assert_has_calls([
        call(tag="weights/fc2/weight", values=ANY, global_step=5),
        call(tag="weights/fc2/bias", values=ANY, global_step=5),
    ], any_order=True)

    with pytest.raises(AssertionError):
        mock_logger.writer.add_histogram.assert_has_calls([
            call(tag="weights/fc1/weight", values=ANY, global_step=5),
            call(tag="weights/fc1/bias", values=ANY, global_step=5),
        ], any_order=True)
    assert mock_logger.writer.add_histogram.call_count == 2


def test_grads_scalar_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        GradsScalarHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    with pytest.raises(TypeError, match="Argument reduction should be callable"):
        GradsScalarHandler(model, reduction=123)

    wrapper = GradsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'GradsScalarHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_scalar_handler(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsScalarHandler(model, reduction=norm_mock, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        norm_mock.reset_mock()

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        mock_logger.writer.add_scalar.assert_has_calls([
            call(tag_prefix + "grads_norm/fc1/weight", ANY, 5),
            call(tag_prefix + "grads_norm/fc1/bias", ANY, 5),
            call(tag_prefix + "grads_norm/fc2/weight", ANY, 5),
            call(tag_prefix + "grads_norm/fc2/bias", ANY, 5),
        ], any_order=True)
        assert mock_logger.writer.add_scalar.call_count == 4
        assert norm_mock.call_count == 4

    _test()
    _test(tag="tag")


def test_grads_scalar_handler_frozen_layers(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = GradsScalarHandler(model, reduction=norm_mock)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    norm_mock.reset_mock()

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.writer.add_scalar.assert_has_calls([
        call("grads_norm/fc2/weight", ANY, 5),
        call("grads_norm/fc2/bias", ANY, 5),
    ], any_order=True)

    with pytest.raises(AssertionError):
        mock_logger.writer.add_scalar.assert_has_calls([
            call("grads_norm/fc1/weight", ANY, 5),
            call("grads_norm/fc1/bias", ANY, 5),
        ], any_order=True)
    assert mock_logger.writer.add_scalar.call_count == 2
    assert norm_mock.call_count == 2


def test_grads_hist_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        GradsHistHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'GradsHistHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_hist_handler(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=TensorboardLogger)
        mock_logger.writer = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.writer.add_histogram.call_count == 4
        mock_logger.writer.add_histogram.assert_has_calls([
            call(tag=tag_prefix + "grads/fc1/weight", values=ANY, global_step=5),
            call(tag=tag_prefix + "grads/fc1/bias", values=ANY, global_step=5),
            call(tag=tag_prefix + "grads/fc2/weight", values=ANY, global_step=5),
            call(tag=tag_prefix + "grads/fc2/bias", values=ANY, global_step=5),
        ], any_order=True)

    _test()
    _test(tag="tag")


def test_grads_hist_frozen_layers(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=True)

    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.writer.add_histogram.call_count == 2
    mock_logger.writer.add_histogram.assert_has_calls([
        call(tag="grads/fc2/weight", values=ANY, global_step=5),
        call(tag="grads/fc2/bias", values=ANY, global_step=5),
    ], any_order=True)

    with pytest.raises(AssertionError):
        mock_logger.writer.add_histogram.assert_has_calls([
            call(tag="grads/fc1/weight", values=ANY, global_step=5),
            call(tag="grads/fc1/bias", values=ANY, global_step=5),
        ], any_order=True)


def test_integration(dirname):

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)

    tb_logger = TensorboardLogger(log_dir=dirname)

    def dummy_handler(engine, logger, event_name):
        global_step = engine.state.get_event_attrib_value(event_name)
        logger.writer.add_scalar("test_value", global_step, global_step)

    tb_logger.attach(trainer,
                     log_handler=dummy_handler,
                     event_name=Events.EPOCH_COMPLETED)

    trainer.run(data, max_epochs=n_epochs)
    tb_logger.close()

    # Check if event files are present
    written_files = os.listdir(dirname)
    written_files = [f for f in written_files if "tfevents" in f]
    assert len(written_files) > 0


def test_integration_as_context_manager(dirname):

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    with TensorboardLogger(log_dir=dirname) as tb_logger:

        trainer = Engine(update_fn)

        def dummy_handler(engine, logger, event_name):
            global_step = engine.state.get_event_attrib_value(event_name)
            logger.writer.add_scalar("test_value", global_step, global_step)

        tb_logger.attach(trainer,
                         log_handler=dummy_handler,
                         event_name=Events.EPOCH_COMPLETED)

        trainer.run(data, max_epochs=n_epochs)

    # Check if event files are present
    written_files = os.listdir(dirname)
    written_files = [f for f in written_files if "tfevents" in f]
    assert len(written_files) > 0


@pytest.fixture
def no_site_packages():
    import sys
    tensorboardX_module = sys.modules['tensorboardX']
    del sys.modules['tensorboardX']
    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    sys.modules['tensorboardX'] = tensorboardX_module


def test_no_tensorboardX(dirname, no_site_packages):

    with pytest.raises(RuntimeError, match=r"This contrib module requires tensorboardX to be installed"):
        TensorboardLogger(log_dir=dirname)


@pytest.fixture
def mock_tb_module():
    import sys
    import types

    module_name = 'tensorboardX'
    tb_module = types.ModuleType(module_name)
    prev_tb_module = sys.modules[module_name]
    sys.modules[module_name] = tb_module

    yield tb_module
    sys.modules[module_name] = prev_tb_module


def test_init_typeerror_exception(mock_tb_module):

    def side_effect(*args, **kwargs):
        raise TypeError("a problem")

    mock_tb_module.SummaryWriter = Mock(name='tensorboardX.SummaryWriter', side_effect=side_effect)

    with pytest.raises(TypeError, match=r'a problem'):
        TensorboardLogger(log_dir=None)
