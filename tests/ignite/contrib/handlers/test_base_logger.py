import math
from unittest.mock import MagicMock

import pytest
import torch

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler
from ignite.engine import Engine, Events, State
from tests.ignite.contrib.handlers import MockFP16DeepSpeedZeroOptimizer


class DummyOutputHandler(BaseOutputHandler):
    def __call__(self, *args, **kwargs):
        pass


class DummyOptParamsHandler(BaseOptimizerParamsHandler):
    def __call__(self, engine, logger, event_name, **kwargs):
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        return params


class DummyLogger(BaseLogger):
    def _create_output_handler(self, *args, **kwargs):
        return DummyOutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args, **kwargs):
        return DummyOptParamsHandler(*args, **kwargs)


def test_base_output_handler_wrong_setup():

    with pytest.raises(TypeError, match="metric_names should be either a list or equal 'all'"):
        DummyOutputHandler("tag", metric_names="abc", output_transform=None)

    with pytest.raises(TypeError, match="output_transform should be a function"):
        DummyOutputHandler("tag", metric_names=None, output_transform="abc")

    with pytest.raises(ValueError, match="Either metric_names or output_transform should be defined"):
        DummyOutputHandler("tag", None, None)

    with pytest.raises(TypeError, match="global_step_transform should be a function"):
        DummyOutputHandler("tag", metric_names=["loss"], global_step_transform="abc")

    with pytest.raises(TypeError, match=r"Argument optimizer should be torch.optim.Optimizer"):
        DummyOptParamsHandler({}, "lr")


def test_base_output_handler_setup_output_metrics():

    engine = Engine(lambda engine, batch: None)
    true_metrics = {"a": 0, "b": 1}
    engine.state = State(metrics=true_metrics)
    engine.state.output = 12345

    # Only metric_names
    handler = DummyOutputHandler("tag", metric_names=["a", "b"], output_transform=None)
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == true_metrics

    # Only metric_names with a warning
    handler = DummyOutputHandler("tag", metric_names=["a", "c"], output_transform=None)
    with pytest.warns(UserWarning):
        metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == {"a": 0}

    # Only output as "output"
    handler = DummyOutputHandler("tag", metric_names=None, output_transform=lambda x: x)
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == {"output": engine.state.output}

    # Only output as "loss"
    handler = DummyOutputHandler("tag", metric_names=None, output_transform=lambda x: {"loss": x})
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == {"loss": engine.state.output}

    # Metrics and output
    handler = DummyOutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == {"a": 0, "b": 1, "loss": engine.state.output}

    # All metrics
    handler = DummyOutputHandler("tag", metric_names="all", output_transform=None)
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == true_metrics


def test_opt_params_handler_on_non_torch_optimizers():
    tensor = torch.zeros([1], requires_grad=True)
    base_optimizer = torch.optim.SGD([tensor], lr=0.1234)
    optimizer = MockFP16DeepSpeedZeroOptimizer(base_optimizer)
    handler = DummyOptParamsHandler(optimizer=optimizer, param_name="lr")
    res = handler(engine=None, logger=None, event_name=None)
    assert isinstance(res, dict)
    assert "lr/group_0" in res and res["lr/group_0"] == 0.1234


def test_attach():

    n_epochs = 5
    data = list(range(50))

    def _test(event, n_calls):

        losses = torch.rand(n_epochs * len(data))
        losses_iter = iter(losses)

        def update_fn(engine, batch):
            return next(losses_iter)

        trainer = Engine(update_fn)

        logger = DummyLogger()

        mock_log_handler = MagicMock()

        logger.attach(trainer, log_handler=mock_log_handler, event_name=event)

        trainer.run(data, max_epochs=n_epochs)

        mock_log_handler.assert_called_with(trainer, logger, event)
        assert mock_log_handler.call_count == n_calls

    _test(Events.ITERATION_STARTED, len(data) * n_epochs)
    _test(Events.ITERATION_COMPLETED, len(data) * n_epochs)
    _test(Events.EPOCH_STARTED, n_epochs)
    _test(Events.EPOCH_COMPLETED, n_epochs)
    _test(Events.STARTED, 1)
    _test(Events.COMPLETED, 1)

    _test(Events.ITERATION_STARTED(every=10), len(data) // 10 * n_epochs)


def test_attach_wrong_event_name():

    trainer = Engine(lambda b, e: None)
    logger = DummyLogger()
    mock_log_handler = MagicMock()

    with pytest.raises(RuntimeError, match="Unknown event name"):
        logger.attach(trainer, log_handler=mock_log_handler, event_name="unknown")


def test_attach_on_custom_event():
    n_epochs = 10
    data = list(range(150))

    def _test(event, n_calls, cpe):

        losses = torch.rand(n_epochs * len(data))
        losses_iter = iter(losses)

        def update_fn(engine, batch):
            return next(losses_iter)

        trainer = Engine(update_fn)
        cpe.attach(trainer)

        logger = DummyLogger()

        mock_log_handler = MagicMock()

        logger.attach(trainer, log_handler=mock_log_handler, event_name=event)

        trainer.run(data, max_epochs=n_epochs)

        mock_log_handler.assert_called_with(trainer, logger, event)
        assert mock_log_handler.call_count == n_calls


def test_as_context_manager():

    n_epochs = 5
    data = list(range(50))

    class _DummyLogger(DummyLogger):
        def __init__(self, writer):
            self.writer = writer

        def close(self):
            self.writer.close()

    def _test(event, n_calls):
        global close_counter
        close_counter = 0

        losses = torch.rand(n_epochs * len(data))
        losses_iter = iter(losses)

        def update_fn(engine, batch):
            return next(losses_iter)

        writer = MagicMock()
        writer.close = MagicMock()

        with _DummyLogger(writer) as logger:
            assert isinstance(logger, _DummyLogger)

            trainer = Engine(update_fn)
            mock_log_handler = MagicMock()

            logger.attach(trainer, log_handler=mock_log_handler, event_name=event)

            trainer.run(data, max_epochs=n_epochs)

            mock_log_handler.assert_called_with(trainer, logger, event)
            assert mock_log_handler.call_count == n_calls

        writer.close.assert_called_once_with()

    _test(Events.ITERATION_STARTED, len(data) * n_epochs)
    _test(Events.ITERATION_COMPLETED, len(data) * n_epochs)
    _test(Events.EPOCH_STARTED, n_epochs)
    _test(Events.EPOCH_COMPLETED, n_epochs)
    _test(Events.STARTED, 1)
    _test(Events.COMPLETED, 1)

    _test(Events.ITERATION_STARTED(every=10), len(data) // 10 * n_epochs)
