import tempfile
import shutil

import torch
import pytest

from mock import MagicMock, call, ANY

from ignite.engine import Engine, Events, State
from ignite.contrib.handlers.visdom_logger import *
from ignite.contrib.handlers.visdom_logger import _DummyExecutor, _BaseVisDrawer


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def visdom_server():

    import time
    import subprocess

    from visdom.server import download_scripts
    download_scripts()

    hostname = "localhost"
    port = 8098
    p = subprocess.Popen("visdom --hostname {} -port {}".format(hostname, port), shell=True)
    time.sleep(5)
    yield (hostname, port)
    p.terminate()


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
    mock_logger.executor = _DummyExecutor()
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

    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()
    mock_logger.executor = _DummyExecutor()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert len(wrapper.windows) == 1 and "generator/lr/group_0" in wrapper.windows
    assert wrapper.windows["generator/lr/group_0"]['win'] is not None

    mock_logger.vis.line.assert_called_once_with(
        X=[123, ], Y=[0.01, ], env=mock_logger.vis.env,
        win=None, update=None,
        opts=wrapper.windows['generator/lr/group_0']['opts'],
        name="generator/lr/group_0"
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
    mock_logger.executor = _DummyExecutor()

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
    mock_logger.executor = _DummyExecutor()

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
    mock_logger.executor = _DummyExecutor()

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
    mock_logger.executor = _DummyExecutor()

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
    mock_logger.executor = _DummyExecutor()

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

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()
    mock_logger.executor = _DummyExecutor()

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


def test_output_handler_both(dirname):

    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()
    mock_logger.executor = _DummyExecutor()

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


def test_output_handler_with_wrong_global_step_transform_output():
    def global_step_transform(*args, **kwargs):
        return 'a'

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()
    mock_logger.executor = _DummyExecutor()

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
    mock_logger = MagicMock(spec=VisdomLogger)
    mock_logger.vis = MagicMock()
    mock_logger.executor = _DummyExecutor()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.vis.line.call_count == 1
    assert len(wrapper.windows) == 1 and "tag/loss" in wrapper.windows
    assert wrapper.windows["tag/loss"]['win'] is not None

    mock_logger.vis.line.assert_has_calls([
        call(X=[10, ], Y=[12345, ], env=mock_logger.vis.env,
             win=None, update=None,
             opts=wrapper.windows['tag/loss']['opts'], name="tag/loss")])


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

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsScalarHandler(model, tag=tag)
        mock_logger = MagicMock(spec=VisdomLogger)
        mock_logger.vis = MagicMock()
        mock_logger.executor = _DummyExecutor()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.vis.line.call_count == 4
        mock_logger.vis.line.assert_has_calls([
            call(X=[5, ], Y=[0.0, ], env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "weights_norm/fc1/weight"]['opts'],
                 name=tag_prefix + "weights_norm/fc1/weight"),
            call(X=[5, ], Y=[0.0, ], env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "weights_norm/fc1/bias"]['opts'],
                 name=tag_prefix + "weights_norm/fc1/bias"),

            call(X=[5, ], Y=[12.0, ], env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "weights_norm/fc2/weight"]['opts'],
                 name=tag_prefix + "weights_norm/fc2/weight"),
            call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "weights_norm/fc2/bias"]['opts'],
                 name=tag_prefix + "weights_norm/fc2/bias"),
        ], any_order=True)

    _test()
    _test(tag="tag")


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
    mock_logger.executor = _DummyExecutor()

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

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsScalarHandler(model, reduction=norm, tag=tag)
        mock_logger = MagicMock(spec=VisdomLogger)
        mock_logger.vis = MagicMock()
        mock_logger.executor = _DummyExecutor()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = "{}/".format(tag) if tag else ""

        assert mock_logger.vis.line.call_count == 4
        mock_logger.vis.line.assert_has_calls([
            call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "grads_norm/fc1/weight"]['opts'],
                 name=tag_prefix + "grads_norm/fc1/weight"),
            call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "grads_norm/fc1/bias"]['opts'],
                 name=tag_prefix + "grads_norm/fc1/bias"),

            call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "grads_norm/fc2/weight"]['opts'],
                 name=tag_prefix + "grads_norm/fc2/weight"),
            call(X=[5, ], Y=ANY, env=mock_logger.vis.env,
                 win=None, update=None,
                 opts=wrapper.windows[tag_prefix + "grads_norm/fc2/bias"]['opts'],
                 name=tag_prefix + "grads_norm/fc2/bias"),
        ], any_order=True)

    _test()
    _test(tag="tag")


def test_integration_no_server():

    with pytest.raises(RuntimeError, match="Failed to connect to Visdom server"):
        VisdomLogger()


def test_logger_init_hostname_port(visdom_server):
    # Explicit hostname, port
    vd_logger = VisdomLogger(server=visdom_server[0], port=visdom_server[1], num_workers=0)
    assert "main" in vd_logger.vis.get_env_list()


def test_logger_init_env_vars(visdom_server):
    # As env vars
    import os
    os.environ['VISDOM_SERVER_URL'] = visdom_server[0]
    os.environ['VISDOM_PORT'] = str(visdom_server[1])
    vd_logger = VisdomLogger(server=visdom_server[0], port=visdom_server[1], num_workers=0)
    assert "main" in vd_logger.vis.get_env_list()


def _parse_content(content):
    import json
    return json.loads(content)


def test_integration_no_executor(visdom_server):
    vd_logger = VisdomLogger(server=visdom_server[0], port=visdom_server[1], num_workers=0)

    # close all windows in 'main' environment
    vd_logger.vis.close()

    n_epochs = 3
    data = list(range(10))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)
    output_handler = OutputHandler(tag="training", output_transform=lambda x: {'loss': x})
    vd_logger.attach(trainer,
                     log_handler=output_handler,
                     event_name=Events.ITERATION_COMPLETED)

    trainer.run(data, max_epochs=n_epochs)

    assert len(output_handler.windows) == 1
    assert "training/loss" in output_handler.windows
    win_name = output_handler.windows['training/loss']['win']
    data = vd_logger.vis.get_window_data(win=win_name)
    data = _parse_content(data)
    assert "content" in data and "data" in data["content"]
    data = data["content"]["data"][0]
    assert "x" in data and "y" in data
    x_vals, y_vals = data['x'], data['y']
    assert all([int(x) == x_true for x, x_true in zip(x_vals, list(range(1, n_epochs * len(data) + 1)))])
    assert all([y == y_true for y, y_true in zip(y_vals, losses)])


def test_integration_with_executor(visdom_server):
    vd_logger = VisdomLogger(server=visdom_server[0], port=visdom_server[1], num_workers=1)

    # close all windows in 'main' environment
    vd_logger.vis.close()

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)
    output_handler = OutputHandler(tag="training", output_transform=lambda x: {'loss': x})
    vd_logger.attach(trainer,
                     log_handler=output_handler,
                     event_name=Events.ITERATION_COMPLETED)

    trainer.run(data, max_epochs=n_epochs)

    assert len(output_handler.windows) == 1
    assert "training/loss" in output_handler.windows
    win_name = output_handler.windows['training/loss']['win']
    data = vd_logger.vis.get_window_data(win=win_name)
    data = _parse_content(data)
    assert "content" in data and "data" in data["content"]
    data = data["content"]["data"][0]
    assert "x" in data and "y" in data
    x_vals, y_vals = data['x'], data['y']
    assert all([int(x) == x_true for x, x_true in zip(x_vals, list(range(1, n_epochs * len(data) + 1)))])
    assert all([y == y_true for y, y_true in zip(y_vals, losses)])

    vd_logger.close()


def test_integration_with_executor_as_context_manager(visdom_server):

    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    with VisdomLogger(server=visdom_server[0], port=visdom_server[1], num_workers=1) as vd_logger:

        # close all windows in 'main' environment
        vd_logger.vis.close()

        trainer = Engine(update_fn)
        output_handler = OutputHandler(tag="training", output_transform=lambda x: {'loss': x})
        vd_logger.attach(trainer,
                         log_handler=output_handler,
                         event_name=Events.ITERATION_COMPLETED)

        trainer.run(data, max_epochs=n_epochs)

        assert len(output_handler.windows) == 1
        assert "training/loss" in output_handler.windows
        win_name = output_handler.windows['training/loss']['win']
        data = vd_logger.vis.get_window_data(win=win_name)
        data = _parse_content(data)
        assert "content" in data and "data" in data["content"]
        data = data["content"]["data"][0]
        assert "x" in data and "y" in data
        x_vals, y_vals = data['x'], data['y']
        assert all([int(x) == x_true for x, x_true in zip(x_vals, list(range(1, n_epochs * len(data) + 1)))])
        assert all([y == y_true for y, y_true in zip(y_vals, losses)])


@pytest.fixture
def no_site_packages():
    import sys
    plx_module = sys.modules['visdom']
    del sys.modules['visdom']
    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    sys.modules['visdom'] = plx_module


def test_no_visdom(no_site_packages):

    with pytest.raises(RuntimeError, match=r"This contrib module requires visdom package"):
        VisdomLogger()
