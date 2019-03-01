import os
import tempfile
import shutil

import pytest

from mock import MagicMock, call

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine, Events, State
from ignite.contrib.handlers.tensorboard_logger import *


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.mark.skipif("dev" in torch.__version__, reason="TensorboardX fails on Pytorch 1.0.0.dev20190301")
def test_log_graph(dirname):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=-1)

    tb_logger = TensorboardLogger(log_dir=dirname)

    model = Net()
    x = torch.rand(2, 1, 28, 28)
    tb_logger.log_graph(model, x)
    tb_logger = None

    files = [f for f in os.listdir(dirname)]
    assert len(files) >= 1 and "events.out.tfevents" in files[0]


def test_attach(dirname):

    n_epochs = 5
    data = list(range(50))

    def _test(event, event_name, n_calls):

        losses = torch.rand(n_epochs * len(data))
        losses_iter = iter(losses)

        def update_fn(engine, batch):
            return next(losses_iter)

        trainer = Engine(update_fn)

        tb_logger = TensorboardLogger(log_dir=dirname)

        mock_log_handler = MagicMock()

        tb_logger.attach(trainer,
                         log_handler=mock_log_handler,
                         event_name=event)

        trainer.run(data, max_epochs=n_epochs)

        mock_log_handler.assert_called_with(trainer, tb_logger.writer, event_name)
        assert mock_log_handler.call_count == n_calls

    _test(Events.ITERATION_STARTED, "iteration", len(data) * n_epochs)
    _test(Events.ITERATION_COMPLETED, "iteration", len(data) * n_epochs)
    _test(Events.EPOCH_STARTED, "epoch", n_epochs)
    _test(Events.EPOCH_COMPLETED, "epoch", n_epochs)
    _test(Events.STARTED, "epoch", 1)
    _test(Events.COMPLETED, "epoch", 1)


def test_output_handler_output_transform(dirname):

    wrapper = output_handler("tag", output_transform=lambda x: x)
    mock_writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_writer, state_attr="iteration")

    mock_writer.add_scalar.assert_called_once_with("tag/output", 12345, 123)

    wrapper = output_handler("another_tag", output_transform=lambda x: {"loss": x})
    mock_writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_writer, state_attr="iteration")

    mock_writer.add_scalar.assert_called_once_with("another_tag/loss", 12345, 123)


def test_output_handler_metric_names(dirname):

    wrapper = output_handler("tag", metric_names=["a", "b"])
    mock_writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_writer, state_attr="iteration")

    assert mock_writer.add_scalar.call_count == 2
    mock_writer.add_scalar.assert_has_calls([
        call("tag/a", 12.23, 5),
        call("tag/b", 23.45, 5),
    ], any_order=True)


def test_output_handler_both(dirname):

    wrapper = output_handler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_writer = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_writer, state_attr="epoch")

    assert mock_writer.add_scalar.call_count == 3
    mock_writer.add_scalar.assert_has_calls([
        call("tag/a", 12.23, 5),
        call("tag/b", 23.45, 5),
        call("tag/loss", 12345, 5)
    ], any_order=True)
