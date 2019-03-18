import os
import tempfile
import shutil
import math

import pytest

from mock import MagicMock, call, ANY

import torch

from ignite.engine import Engine, Events, State
from ignite.contrib.handlers.visdom_logger import *


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
    with pytest.raises(RuntimeError, match="Handler 'OptimizerParamsHandler' works only with VisdomLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():

    optimizer = torch.optim.SGD([torch.Tensor(0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    # mock_logger.vis.line.assert_called_once_with("lr/group_0", 0.01, 123)
    assert len(wrapper.windows) == 1 and "lr/group_0" in wrapper.windows
    assert wrapper.windows["lr/group_0"]['win'] is not None

    mock_logger.vis.line.assert_called_once_with(
        X=[123, ], Y=[0.01, ], env=mock_logger.vis.env,
        win=None, update=None,
        opts=wrapper.windows['lr/group_0']['opts'],
        name="lr/group_0"
    )


def test_output_handler_with_wrong_logger_type():

    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'OutputHandler' works only with VisdomLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform(dirname):

    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert len(wrapper.windows) == 1 and "tag/output" in wrapper.windows
    assert wrapper.windows["tag/output"]['win'] is not None

    mock_logger.vis.line.assert_called_once_with(
        X=[123, ], Y=[12345, ], env=mock_logger.vis.env,
        win=None, update=None,
        opts=wrapper.windows['tag/output']['opts'],
        name="tag/output"
    )

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert len(wrapper.windows) == 1 and "another_tag/loss" in wrapper.windows
    assert wrapper.windows["another_tag/loss"]['win'] is not None

    mock_logger.vis.line.assert_called_once_with(
        X=[123, ], Y=[12345, ], env=mock_logger.vis.env,
        win=None, update=None,
        opts=wrapper.windows['another_tag/loss']['opts'],
        name="another_tag/loss"
    )


def test_output_handler_metric_names(dirname):

    wrapper = OutputHandler("tag", metric_names=["a", "b"])
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert len(wrapper.windows) == 2 and \
        "tag/a" in wrapper.windows and "tag/b" in wrapper.windows
    assert wrapper.windows["tag/a"]['win'] is not None
    assert wrapper.windows["tag/b"]['win'] is not None

    assert mock_logger.vis.line.call_count == 2
    mock_logger.vis.line.assert_has_calls([
        call(X=[5, ], Y=[12.23, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/a']['opts'], name="tag/a"),
        call(X=[5, ], Y=[23.45, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/b']['opts'], name="tag/b"),
    ], any_order=True)

    wrapper = OutputHandler("tag", metric_names=["a", ])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.Tensor([0.0, 1.0, 2.0, 3.0])})
    mock_engine.state.iteration = 5

    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert len(wrapper.windows) == 4 and \
        all(["tag/a/{}".format(i) in wrapper.windows for i in range(4)])
    assert wrapper.windows["tag/a/0"]['win'] is not None
    assert wrapper.windows["tag/a/1"]['win'] is not None
    assert wrapper.windows["tag/a/2"]['win'] is not None
    assert wrapper.windows["tag/a/3"]['win'] is not None

    assert mock_logger.vis.line.call_count == 4
    mock_logger.vis.line.assert_has_calls([
        call(X=[5, ], Y=[0.0, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/a/0']['opts'], name="tag/a/0"),
        call(X=[5, ], Y=[1.0, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/a/1']['opts'], name="tag/a/1"),
        call(X=[5, ], Y=[2.0, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/a/2']['opts'], name="tag/a/2"),
        call(X=[5, ], Y=[3.0, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/a/3']['opts'], name="tag/a/3"),
    ], any_order=True)

    wrapper = OutputHandler("tag", metric_names=["a", "c"])

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 55.56, "c": "Some text"})
    mock_engine.state.iteration = 7

    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    with pytest.warns(UserWarning):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert len(wrapper.windows) == 1 and "tag/a" in wrapper.windows
    assert wrapper.windows["tag/a"]['win'] is not None

    assert mock_logger.vis.line.call_count == 1
    mock_logger.vis.line.assert_has_calls([
        call(X=[7, ], Y=[55.56, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/a']['opts'], name="tag/a"),
    ], any_order=True)


def test_output_handler_both(dirname):

    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.vis.line.call_count == 3
    assert len(wrapper.windows) == 3 and \
        "tag/a" in wrapper.windows and "tag/b" in wrapper.windows and "tag/loss" in wrapper.windows
    assert wrapper.windows["tag/a"]['win'] is not None
    assert wrapper.windows["tag/b"]['win'] is not None
    assert wrapper.windows["tag/loss"]['win'] is not None

    mock_logger.vis.line.assert_has_calls([
        call(X=[5, ], Y=[12.23, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/a']['opts'], name="tag/a"),
        call(X=[5, ], Y=[23.45, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/b']['opts'], name="tag/b"),
        call(X=[5, ], Y=[12345, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/loss']['opts'], name="tag/loss"),
    ], any_order=True)

    mock_engine.state.epoch = 6
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.vis.line.call_count == 6
    assert len(wrapper.windows) == 3 and \
        "tag/a" in wrapper.windows and "tag/b" in wrapper.windows and "tag/loss" in wrapper.windows
    assert wrapper.windows["tag/a"]['win'] is not None
    assert wrapper.windows["tag/b"]['win'] is not None
    assert wrapper.windows["tag/loss"]['win'] is not None

    mock_logger.vis.line.assert_has_calls([
        call(X=[6, ], Y=[12.23, ], env=mock_logger.vis.env,
             win=wrapper.windows["tag/a"]['win'], update='append',
             opts=wrapper.windows['tag/a']['opts'], name="tag/a"),
        call(X=[6, ], Y=[23.45, ], env=mock_logger.vis.env,
             win=wrapper.windows["tag/b"]['win'], update='append',
             opts=wrapper.windows['tag/b']['opts'], name="tag/b"),
        call(X=[6, ], Y=[12345, ], env=mock_logger.vis.env,
             win=wrapper.windows["tag/loss"]['win'], update='append',
             opts=wrapper.windows['tag/loss']['opts'], name="tag/loss"),
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
    with pytest.raises(RuntimeError, match="Handler 'WeightsScalarHandler' works only with VisdomLogger"):
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
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.vis.line.call_count == 4
    mock_logger.vis.line.assert_has_calls([
        call(X=[5, ], Y=[0.0, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc1/weight"]['opts'], name="weights_norm/fc1/weight"),
        call(X=[5, ], Y=[0.0, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc1/bias"]['opts'], name="weights_norm/fc1/bias"),

        call(X=[5, ], Y=[12.0, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc2/weight"]['opts'], name="weights_norm/fc2/weight"),
        call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc2/bias"]['opts'], name="weights_norm/fc2/bias"),

    ], any_order=True)


def test_weights_scalar_handler_custom_reduction():

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
        return 12.34

    wrapper = WeightsScalarHandler(model, reduction=norm, show_legend=True)
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.vis.line.call_count == 4
    mock_logger.vis.line.assert_has_calls([
        call(X=[5, ], Y=[12.34, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc1/weight"]['opts'], name="weights_norm/fc1/weight"),
        call(X=[5, ], Y=[12.34, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc1/bias"]['opts'], name="weights_norm/fc1/bias"),

        call(X=[5, ], Y=[12.34, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc2/weight"]['opts'], name="weights_norm/fc2/weight"),
        call(X=[5, ], Y=[12.34, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["weights_norm/fc2/bias"]['opts'], name="weights_norm/fc2/bias"),

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
    with pytest.raises(RuntimeError, match="Handler 'GradsScalarHandler' works only with VisdomLogger"):
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
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.vis.line.call_count == 4

    mock_logger.vis.line.assert_has_calls([
        call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["grads_norm/fc1/weight"]['opts'], name="grads_norm/fc1/weight"),
        call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["grads_norm/fc1/bias"]['opts'], name="grads_norm/fc1/bias"),

        call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["grads_norm/fc2/weight"]['opts'], name="grads_norm/fc2/weight"),
        call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows["grads_norm/fc2/bias"]['opts'], name="grads_norm/fc2/bias"),

    ], any_order=True)


def test_intergration_no_server():

    with pytest.raises(RuntimeError, match="Failed to connect to Visdom server"):
        VisdomLogger()
