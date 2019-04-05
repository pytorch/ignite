import os
import tempfile
import shutil
import math

import pytest

from mock import MagicMock, call, ANY

import torch

from ignite.engine import Engine, Events, State
from ignite.contrib.handlers.tensorboard_logger import *


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


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


def test_weights_scalar_handler():

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(12, 12)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.fill_(1.0)
            self.fc2.bias.data.fill_(1.0)

    model = DummyModel()

    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.writer.add_scalar.call_count == 4
    mock_logger.writer.add_scalar.assert_has_calls([
        call("weights_norm/fc1/weight", 0.0, 5),
        call("weights_norm/fc1/bias", 0.0, 5),
        call("weights_norm/fc2/weight", 12.0, 5),
        call("weights_norm/fc2/bias", math.sqrt(12.0), 5),
    ], any_order=True)


def test_weights_hist_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        WeightsHistHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'WeightsHistHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_hist_handler():

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(12, 12)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.fill_(1.0)
            self.fc2.bias.data.fill_(1.0)

    model = DummyModel()

    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.writer.add_histogram.call_count == 4
    mock_logger.writer.add_histogram.assert_has_calls([
        call(tag="weights/fc1/weight", values=ANY, global_step=5),
        call(tag="weights/fc1/bias", values=ANY, global_step=5),
        call(tag="weights/fc2/weight", values=ANY, global_step=5),
        call(tag="weights/fc2/bias", values=ANY, global_step=5),
    ], any_order=True)


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


def test_grads_scalar_handler():

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(12, 12)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.fill_(1.0)
            self.fc2.bias.data.fill_(1.0)

    model = DummyModel()

    def norm(x):
        return 0.0

    wrapper = GradsScalarHandler(model, reduction=norm)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.writer.add_scalar.call_count == 4

    mock_logger.writer.add_scalar.assert_has_calls([
        call("grads_norm/fc1/weight", ANY, 5),
        call("grads_norm/fc1/bias", ANY, 5),
        call("grads_norm/fc2/weight", ANY, 5),
        call("grads_norm/fc2/bias", ANY, 5),
    ], any_order=True)


def test_grads_hist_handler_wrong_setup():

    with pytest.raises(TypeError, match="Argument model should be of type torch.nn.Module"):
        GradsHistHandler(None)

    model = MagicMock(spec=torch.nn.Module)
    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'GradsHistHandler' works only with TensorboardLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_hist_handler():

    class DummyModel(torch.nn.Module):

        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(12, 12)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.fill_(1.0)
            self.fc2.bias.data.fill_(1.0)

    model = DummyModel()

    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock(spec=TensorboardLogger)
    mock_logger.writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    model.fc1.weight.grad = torch.zeros_like(model.fc1.weight)
    model.fc1.bias.grad = torch.zeros_like(model.fc1.bias)

    model.fc2.weight.grad = torch.zeros_like(model.fc2.weight)
    model.fc2.bias.grad = torch.zeros_like(model.fc2.bias)

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.writer.add_histogram.call_count == 4
    mock_logger.writer.add_histogram.assert_has_calls([
        call(tag="grads/fc1/weight", values=ANY, global_step=5),
        call(tag="grads/fc1/bias", values=ANY, global_step=5),
        call(tag="grads/fc2/weight", values=ANY, global_step=5),
        call(tag="grads/fc2/bias", values=ANY, global_step=5),
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
