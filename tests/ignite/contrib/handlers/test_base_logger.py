import math
import torch

from ignite.engine import Engine, State, Events
from ignite.engine.engine import EventWithFilter
from ignite.contrib.handlers.base_logger import BaseLogger, BaseOutputHandler, global_step_from_engine
from ignite.contrib.handlers import CustomPeriodicEvent

import pytest

from unittest.mock import MagicMock


class DummyLogger(BaseLogger):
    pass


class DummyOutputHandler(BaseOutputHandler):

    def __call__(self, *args, **kwargs):
        pass


def test_base_output_handler_wrong_setup():

    with pytest.raises(TypeError, match="metric_names should be either a list or equal 'all'"):
        DummyOutputHandler("tag", metric_names="abc", output_transform=None)

    with pytest.raises(TypeError, match="output_transform should be a function"):
        DummyOutputHandler("tag", metric_names=None, output_transform="abc")

    with pytest.raises(ValueError, match="Either metric_names or output_transform should be defined"):
        DummyOutputHandler("tag", None, None)

    with pytest.raises(TypeError, match="Argument another_engine should be of type Engine"):
        DummyOutputHandler("tag", ["a", "b"], None, another_engine=123)

    with pytest.raises(TypeError, match="global_step_transform should be a function"):
        DummyOutputHandler("tag", metric_names=["loss"], global_step_transform="abc")


def test_base_output_handler_with_another_engine():
    engine = Engine(lambda engine, batch: None)
    true_metrics = {"a": 0, "b": 1}
    engine.state = State(metrics=true_metrics)
    engine.state.output = 12345

    with pytest.warns(DeprecationWarning, match="Use of another_engine is deprecated"):
        handler = DummyOutputHandler("tag", metric_names=['a', 'b'], output_transform=None, another_engine=engine)


def test_base_output_handler_setup_output_metrics():

    engine = Engine(lambda engine, batch: None)
    true_metrics = {"a": 0, "b": 1}
    engine.state = State(metrics=true_metrics)
    engine.state.output = 12345

    # Only metric_names
    handler = DummyOutputHandler("tag", metric_names=['a', 'b'], output_transform=None)
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == true_metrics

    # Only metric_names with a warning
    handler = DummyOutputHandler("tag", metric_names=['a', 'c'], output_transform=None)
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
    handler = DummyOutputHandler("tag", metric_names=['a', 'b'], output_transform=lambda x: {"loss": x})
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == {"a": 0, "b": 1, "loss": engine.state.output}

    # All metrics
    handler = DummyOutputHandler("tag", metric_names="all", output_transform=None)
    metrics = handler._setup_output_metrics(engine=engine)
    assert metrics == true_metrics


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

        logger.attach(trainer,
                      log_handler=mock_log_handler,
                      event_name=event)

        trainer.run(data, max_epochs=n_epochs)

        if isinstance(event, EventWithFilter):
            event = event.event

        mock_log_handler.assert_called_with(trainer, logger, event)
        assert mock_log_handler.call_count == n_calls

    _test(Events.ITERATION_STARTED, len(data) * n_epochs)
    _test(Events.ITERATION_COMPLETED, len(data) * n_epochs)
    _test(Events.EPOCH_STARTED, n_epochs)
    _test(Events.EPOCH_COMPLETED, n_epochs)
    _test(Events.STARTED, 1)
    _test(Events.COMPLETED, 1)

    _test(Events.ITERATION_STARTED(every=10), len(data) // 10 * n_epochs)


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

        logger.attach(trainer,
                      log_handler=mock_log_handler,
                      event_name=event)

        trainer.run(data, max_epochs=n_epochs)

        mock_log_handler.assert_called_with(trainer, logger, event)
        assert mock_log_handler.call_count == n_calls

    n_iterations = 10
    cpe1 = CustomPeriodicEvent(n_iterations=n_iterations)
    n = len(data) * n_epochs / n_iterations
    nf = math.floor(n)
    ns = nf + 1 if nf < n else nf
    _test(cpe1.Events.ITERATIONS_10_STARTED, ns, cpe1)
    _test(cpe1.Events.ITERATIONS_10_COMPLETED, nf, cpe1)

    n_iterations = 15
    cpe2 = CustomPeriodicEvent(n_iterations=n_iterations)
    n = len(data) * n_epochs / n_iterations
    nf = math.floor(n)
    ns = nf + 1 if nf < n else nf
    _test(cpe2.Events.ITERATIONS_15_STARTED, ns, cpe2)
    _test(cpe2.Events.ITERATIONS_15_COMPLETED, nf, cpe2)

    n_custom_epochs = 2
    cpe3 = CustomPeriodicEvent(n_epochs=n_custom_epochs)
    n = n_epochs / n_custom_epochs
    nf = math.floor(n)
    ns = nf + 1 if nf < n else nf
    _test(cpe3.Events.EPOCHS_2_STARTED, ns, cpe3)
    _test(cpe3.Events.EPOCHS_2_COMPLETED, nf, cpe3)


def test_as_context_manager():

    n_epochs = 5
    data = list(range(50))

    class _DummyLogger(BaseLogger):

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

            logger.attach(trainer,
                          log_handler=mock_log_handler,
                          event_name=event)

            trainer.run(data, max_epochs=n_epochs)

            if isinstance(event, EventWithFilter):
                event = event.event

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


def test_global_step_from_engine():

    engine = Engine(lambda engine, batch: None)
    engine.state = State()
    engine.state.epoch = 1

    another_engine = Engine(lambda engine, batch: None)
    another_engine.state = State()
    another_engine.state.epoch = 10

    global_step_transform = global_step_from_engine(another_engine)
    res = global_step_transform(engine, Events.EPOCH_COMPLETED)

    assert res == another_engine.state.epoch
