import math
import os
from collections import defaultdict
from unittest.mock import ANY, call, MagicMock, patch

import clearml
import pytest
import torch
from clearml.binding.frameworks import WeightsFileHandler
from clearml.model import Framework

import ignite.distributed as idist

from ignite.engine import Engine, Events, State
from ignite.handlers import Checkpoint
from ignite.handlers.clearml_logger import (
    ClearMLLogger,
    ClearMLSaver,
    global_step_from_engine,
    GradsHistHandler,
    GradsScalarHandler,
    OptimizerParamsHandler,
    OutputHandler,
    WeightsHistHandler,
    WeightsScalarHandler,
)


def test_no_clearml():
    with patch.dict("sys.modules", {"clearml": None, "trains": None}):
        with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires clearml to be installed."):
            ClearMLSaver()

        with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires clearml to be installed."):
            ClearMLLogger()

    with patch.dict("sys.modules", {"clearml.binding.frameworks.tensorflow_bind": None}):
        with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires clearml to be installed."):
            ClearMLLogger()

    with patch.dict("sys.modules", {"clearml.binding.frameworks": None, "trains.binding.frameworks": None}):
        with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires clearml to be installed."):
            ClearMLSaver.__call__(None, {}, "")


def test_optimizer_params_handler_wrong_setup():
    with pytest.raises(TypeError):
        OptimizerParamsHandler(optimizer=None)

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    handler = OptimizerParamsHandler(optimizer=optimizer)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler OptimizerParamsHandler works only with ClearMLLogger"):
        handler(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_optimizer_params():
    optimizer = torch.optim.SGD([torch.tensor(0.0)], lr=0.01)
    wrapper = OptimizerParamsHandler(optimizer=optimizer, param_name="lr")
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()
    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.clearml_logger.report_scalar.assert_called_once_with(iteration=123, series="0", title="lr", value=0.01)

    wrapper = OptimizerParamsHandler(optimizer, param_name="lr", tag="generator")
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.clearml_logger.report_scalar.assert_called_once_with(
        iteration=123, series="0", title="generator/lr", value=0.01
    )


def test_output_handler_with_wrong_logger_type():
    wrapper = OutputHandler("tag", output_transform=lambda x: x)

    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler OutputHandler works only with ClearMLLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_output_handler_output_transform(dirname):
    wrapper = OutputHandler("tag", output_transform=lambda x: x)
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.output = 12345
    mock_engine.state.iteration = 123

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    mock_logger.clearml_logger.report_scalar.assert_called_once_with(
        iteration=123, series="output", title="tag", value=12345
    )

    wrapper = OutputHandler("another_tag", output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)
    mock_logger.clearml_logger.report_scalar.assert_called_once_with(
        iteration=123, series="loss", title="another_tag", value=12345
    )


def test_output_handler_metric_names(dirname):
    wrapper = OutputHandler("tag", metric_names=["a", "b"])
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 2
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
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

    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    with pytest.warns(UserWarning, match=r"Logger output_handler can not log metrics value type"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 1
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="a", iteration=7, value=55.56)], any_order=True
    )

    # all metrics
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 2
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="tag", series="a", iteration=5, value=12.23),
            call(title="tag", series="b", iteration=5, value=23.45),
        ],
        any_order=True,
    )

    # log a torch vector
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    vector = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.33])
    mock_engine.state = State(metrics={"vector": vector})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 5
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [call(title="tag/vector", series=str(i), iteration=5, value=vector[i].item()) for i in range(5)],
        any_order=True,
    )

    # log a torch tensor (ndimension = 0)
    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": torch.tensor(12.23), "b": torch.tensor(23.45), "c": torch.tensor(5.01)})
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 3
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="tag", series="a", iteration=5, value=torch.tensor(12.23).item()),
            call(title="tag", series="b", iteration=5, value=torch.tensor(23.45).item()),
            call(title="tag", series="c", iteration=5, value=torch.tensor(5.01).item()),
        ],
        any_order=True,
    )

    wrapper = OutputHandler("tag", metric_names="all")
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(
        metrics={
            "a": 123,
            "b": {"c": [2.34, {"d": 1}]},
            "c": (22, [33, -5.5], {"e": 32.1}),
        }
    )
    mock_engine.state.iteration = 5

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 7
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="tag", series="a", iteration=5, value=123),
            call(title="tag/b", series="c/0", iteration=5, value=2.34),
            call(title="tag/b", series="c/1/d", iteration=5, value=1),
            call(title="tag/c", series="0", iteration=5, value=22),
            call(title="tag/c", series="1/0", iteration=5, value=33),
            call(title="tag/c", series="1/1", iteration=5, value=-5.5),
            call(title="tag/c", series="2/e", iteration=5, value=32.1),
        ],
        any_order=True,
    )


def test_output_handler_both(dirname):
    wrapper = OutputHandler("tag", metric_names=["a", "b"], output_transform=lambda x: {"loss": x})
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State(metrics={"a": 12.23, "b": 23.45})
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 3
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
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
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

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

    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 1
    mock_engine.state.output = 0.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.clearml_logger.report_scalar.call_count == 1
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="loss", iteration=mock_another_engine.state.epoch, value=mock_engine.state.output)]
    )

    mock_another_engine.state.epoch = 11
    mock_engine.state.output = 1.123

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.clearml_logger.report_scalar.call_count == 2
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="loss", iteration=mock_another_engine.state.epoch, value=mock_engine.state.output)]
    )


def test_output_handler_state_attrs():
    wrapper = OutputHandler("tag", state_attributes=["alpha", "beta", "gamma"])
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.iteration = 5
    mock_engine.state.alpha = 3.899
    mock_engine.state.beta = torch.tensor(12.0)
    mock_engine.state.gamma = torch.tensor([21.0, 6.0])

    wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)

    assert mock_logger.clearml_logger.report_scalar.call_count == 4
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="tag", series="alpha", iteration=5, value=3.899),
            call(title="tag", series="beta", iteration=5, value=12.0),
            call(title="tag/gamma", series="0", iteration=5, value=21.0),
            call(title="tag/gamma", series="1", iteration=5, value=6.0),
        ],
        any_order=True,
    )


def test_output_handler_with_global_step_transform():
    def global_step_transform(*args, **kwargs):
        return 10

    wrapper = OutputHandler("tag", output_transform=lambda x: {"loss": x}, global_step_transform=global_step_transform)
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5
    mock_engine.state.output = 12345

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    assert mock_logger.clearml_logger.report_scalar.call_count == 1
    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [call(title="tag", series="loss", iteration=10, value=12345)]
    )


def test_weights_scalar_handler_wrong_setup():
    model = MagicMock(spec=torch.nn.Module)
    wrapper = WeightsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler WeightsScalarHandler works only with ClearMLLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_scalar_handler(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsScalarHandler(model, tag=tag)
        mock_logger = MagicMock(spec=ClearMLLogger)
        mock_logger.clearml_logger = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

        assert mock_logger.clearml_logger.report_scalar.call_count == 4
        mock_logger.clearml_logger.report_scalar.assert_has_calls(
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


def test_weights_scalar_handler_whitelist(dummy_model_factory):
    model = dummy_model_factory()

    wrapper = WeightsScalarHandler(model, whitelist=["fc2.weight"])
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.clearml_logger.report_scalar.assert_called_once_with(
        title="weights_norm/fc2", value=ANY, series="weight", iteration=mock_engine.state.epoch
    )
    mock_logger.clearml_logger.report_scalar.reset_mock()

    wrapper = WeightsScalarHandler(model, tag="model", whitelist=["fc1"])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="model/weights_norm/fc1", value=ANY, series="weight", iteration=mock_engine.state.epoch),
            call(title="model/weights_norm/fc1", value=ANY, series="bias", iteration=mock_engine.state.epoch),
        ],
        any_order=True,
    )
    assert mock_logger.clearml_logger.report_scalar.call_count == 2
    mock_logger.clearml_logger.report_scalar.reset_mock()

    def weight_selector(n, _):
        return "bias" in n

    wrapper = WeightsScalarHandler(model, tag="model", whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="model/weights_norm/fc1", value=ANY, series="bias", iteration=mock_engine.state.epoch),
            call(title="model/weights_norm/fc2", value=ANY, series="bias", iteration=mock_engine.state.epoch),
        ],
        any_order=True,
    )
    assert mock_logger.clearml_logger.report_scalar.call_count == 2


def test_weights_hist_handler_wrong_setup():
    model = MagicMock(spec=torch.nn.Module)
    wrapper = WeightsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'WeightsHistHandler' works only with ClearMLLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_weights_hist_handler(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = WeightsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=ClearMLLogger)
        mock_logger.grad_helper = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

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


def test_weights_hist_handler_whitelist(dummy_model_factory):
    model = dummy_model_factory()

    wrapper = WeightsHistHandler(model, whitelist=["fc2.weight"])
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.grad_helper = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.grad_helper.add_histogram.assert_called_once_with(
        title="weights_fc2", hist_data=ANY, series="weight", step=5
    )
    mock_logger.grad_helper.add_histogram.reset_mock()

    wrapper = WeightsHistHandler(model, tag="model", whitelist=["fc1"])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.grad_helper.add_histogram.assert_has_calls(
        [
            call(title="model/weights_fc1", hist_data=ANY, series="weight", step=5),
            call(title="model/weights_fc1", hist_data=ANY, series="bias", step=5),
        ],
        any_order=True,
    )
    assert mock_logger.grad_helper.add_histogram.call_count == 2
    mock_logger.grad_helper.add_histogram.reset_mock()

    def weight_selector(n, _):
        return "bias" in n

    wrapper = WeightsHistHandler(model, tag="model", whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.grad_helper.add_histogram.assert_has_calls(
        [
            call(title="model/weights_fc1", hist_data=ANY, series="bias", step=5),
            call(title="model/weights_fc2", hist_data=ANY, series="bias", step=5),
        ],
        any_order=True,
    )
    assert mock_logger.grad_helper.add_histogram.call_count == 2


def test_grads_scalar_handler_wrong_setup():
    model = MagicMock(spec=torch.nn.Module)
    wrapper = GradsScalarHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler GradsScalarHandler works only with ClearMLLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_scalar_handler(dummy_model_factory, norm_mock):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsScalarHandler(model, reduction=norm_mock, tag=tag)
        mock_logger = MagicMock(spec=ClearMLLogger)
        mock_logger.clearml_logger = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5
        norm_mock.reset_mock()

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

        mock_logger.clearml_logger.report_scalar.assert_has_calls(
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
        assert mock_logger.clearml_logger.report_scalar.call_count == 4
        assert norm_mock.call_count == 4

    _test()
    _test(tag="tag")


def test_grads_scalar_handler_whitelist(dummy_model_factory):
    model = dummy_model_factory()

    wrapper = GradsScalarHandler(model, whitelist=["fc2.weight"])
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.clearml_logger = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.clearml_logger.report_scalar.assert_called_once_with(
        title="grads_norm/fc2", value=ANY, series="weight", iteration=mock_engine.state.epoch
    )
    mock_logger.clearml_logger.report_scalar.reset_mock()

    wrapper = GradsScalarHandler(model, tag="model", whitelist=["fc1"])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="model/grads_norm/fc1", value=ANY, series="weight", iteration=mock_engine.state.epoch),
            call(title="model/grads_norm/fc1", value=ANY, series="bias", iteration=mock_engine.state.epoch),
        ],
        any_order=True,
    )
    assert mock_logger.clearml_logger.report_scalar.call_count == 2
    mock_logger.clearml_logger.report_scalar.reset_mock()

    def weight_selector(n, _):
        return "bias" in n

    wrapper = GradsScalarHandler(model, tag="model", whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.clearml_logger.report_scalar.assert_has_calls(
        [
            call(title="model/grads_norm/fc1", value=ANY, series="bias", iteration=mock_engine.state.epoch),
            call(title="model/grads_norm/fc2", value=ANY, series="bias", iteration=mock_engine.state.epoch),
        ],
        any_order=True,
    )
    assert mock_logger.clearml_logger.report_scalar.call_count == 2


def test_grads_hist_handler_wrong_setup():
    model = MagicMock(spec=torch.nn.Module)
    wrapper = GradsHistHandler(model)
    mock_logger = MagicMock()
    mock_engine = MagicMock()
    with pytest.raises(RuntimeError, match="Handler 'GradsHistHandler' works only with ClearMLLogger"):
        wrapper(mock_engine, mock_logger, Events.ITERATION_STARTED)


def test_grads_hist_handler(dummy_model_factory):
    model = dummy_model_factory(with_grads=True, with_frozen_layer=False)

    # define test wrapper to test with and without optional tag
    def _test(tag=None):
        wrapper = GradsHistHandler(model, tag=tag)
        mock_logger = MagicMock(spec=ClearMLLogger)
        mock_logger.grad_helper = MagicMock()

        mock_engine = MagicMock()
        mock_engine.state = State()
        mock_engine.state.epoch = 5

        wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

        tag_prefix = f"{tag}/" if tag else ""

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


def test_grads_hist_handler_whitelist(dummy_model_factory):
    model = dummy_model_factory()

    wrapper = GradsHistHandler(model, whitelist=["fc2.weight"])
    mock_logger = MagicMock(spec=ClearMLLogger)
    mock_logger.grad_helper = MagicMock()

    mock_engine = MagicMock()
    mock_engine.state = State()
    mock_engine.state.epoch = 5

    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)
    mock_logger.grad_helper.add_histogram.assert_called_once_with(
        title="grads_fc2", hist_data=ANY, series="weight", step=5
    )
    mock_logger.grad_helper.reset_mock()

    wrapper = GradsHistHandler(model, tag="model", whitelist=["fc1"])
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.grad_helper.add_histogram.assert_has_calls(
        [
            call(title="model/grads_fc1", hist_data=ANY, series="weight", step=5),
            call(title="model/grads_fc1", hist_data=ANY, series="bias", step=5),
        ],
        any_order=True,
    )
    assert mock_logger.grad_helper.add_histogram.call_count == 2
    mock_logger.grad_helper.reset_mock()

    def weight_selector(n, _):
        return "bias" in n

    wrapper = GradsHistHandler(model, tag="model", whitelist=weight_selector)
    wrapper(mock_engine, mock_logger, Events.EPOCH_STARTED)

    mock_logger.grad_helper.add_histogram.assert_has_calls(
        [
            call(title="model/grads_fc1", hist_data=ANY, series="bias", step=5),
            call(title="model/grads_fc2", hist_data=ANY, series="bias", step=5),
        ],
        any_order=True,
    )
    assert mock_logger.grad_helper.add_histogram.call_count == 2


def test_integration(dirname):
    n_epochs = 5
    data = list(range(50))

    losses = torch.rand(n_epochs * len(data))
    losses_iter = iter(losses)

    def update_fn(engine, batch):
        return next(losses_iter)

    trainer = Engine(update_fn)

    with pytest.warns(UserWarning, match="ClearMLSaver: running in bypass mode"):
        ClearMLLogger.set_bypass_mode(True)
        logger = ClearMLLogger(output_uri=dirname)

        def dummy_handler(engine, logger, event_name):
            global_step = engine.state.get_event_attrib_value(event_name)
            test_value = 0.3  # example
            logger.clearml_logger.report_scalar(title="", series="", value=test_value, iteration=global_step)

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

    with pytest.warns(UserWarning, match="ClearMLSaver: running in bypass mode"):
        ClearMLLogger.set_bypass_mode(True)
        with ClearMLLogger(output_uri=dirname) as clearml_logger:
            trainer = Engine(update_fn)

            def dummy_handler(engine, logger, event_name):
                global_step = engine.state.get_event_attrib_value(event_name)
                test_value = 0.3  # example
                logger.clearml_logger.report_scalar(title="", series="", value=test_value, iteration=global_step)

            clearml_logger.attach(trainer, log_handler=dummy_handler, event_name=Events.EPOCH_COMPLETED)

            trainer.run(data, max_epochs=n_epochs)


def test_clearml_logger_getattr_method(dirname):
    with pytest.warns(UserWarning, match="ClearMLSaver: running in bypass mode"):
        ClearMLLogger.set_bypass_mode(True)

        logger = ClearMLLogger(output_uri=dirname)

        # Create a mock clearml.Logger() object
        mock_logger = MagicMock()
        logger.clearml_logger = mock_logger

        # Test a method called by __getattr__ calls the corresponding method of the mock project.
        logger.report_single_value("accuracy", 0.72)
        mock_logger.report_single_value.assert_called_once_with("accuracy", 0.72)

        # Test a method called by __getattr__ calls the corresponding classmethod of the mock project's class.
        logger.current_logger()
        mock_logger.current_logger.assert_called_once()

        logger.close()


def test_clearml_logger_get_task_bypass(dirname):
    with pytest.warns(UserWarning, match="ClearMLSaver: running in bypass mode"):
        ClearMLLogger.set_bypass_mode(True)

        with ClearMLLogger(output_uri=dirname) as clearml_logger:
            task = clearml_logger.get_task()
            assert isinstance(task, clearml.Task)
            assert task == clearml.Task.current_task()
            task.close()


def test_clearml_disk_saver_integration():
    model = torch.nn.Module()
    to_save_serializable = {"model": model}
    with pytest.warns(UserWarning, match="ClearMLSaver created a temporary checkpoints directory"):
        mock_logger = MagicMock(spec=ClearMLLogger)
        clearml.Task.current_task = MagicMock(spec=clearml.Task)
        clearml_saver = ClearMLSaver(mock_logger)
        clearml.binding.frameworks.WeightsFileHandler.create_output_model = MagicMock()

    checkpoint = Checkpoint(to_save=to_save_serializable, save_handler=clearml_saver, n_saved=1)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)
    checkpoint(trainer)
    trainer.state.iteration = 1
    checkpoint(trainer)
    if clearml_saver._atomic:
        assert clearml.binding.frameworks.WeightsFileHandler.create_output_model.call_count == 2
    else:
        saved_files = list(os.listdir(clearml_saver.dirname))
        assert len(saved_files) == 1
        assert saved_files[0] == "model_1.pt"


def test_clearml_disk_saver_integration_no_logger():
    model = torch.nn.Module()
    to_save_serializable = {"model": model}

    with pytest.warns(UserWarning, match="ClearMLSaver created a temporary checkpoints directory"):
        clearml.Task.current_task = MagicMock(spec=clearml.Task)
        clearml.binding.frameworks.WeightsFileHandler.create_output_model = MagicMock()
        clearml_saver = ClearMLSaver()
        checkpoint = Checkpoint(to_save=to_save_serializable, save_handler=clearml_saver, n_saved=1)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=0, iteration=0)
    checkpoint(trainer)
    trainer.state.iteration = 1
    checkpoint(trainer)

    if clearml_saver._atomic:
        assert clearml.binding.frameworks.WeightsFileHandler.create_output_model.call_count == 2
    else:
        saved_files = list(os.listdir(clearml_saver.dirname))
        assert len(saved_files) == 1
        assert saved_files[0] == "model_1.pt"


def test_clearml_saver_callbacks():
    mock_task = MagicMock(spec=clearml.Task)
    mock_task.name = "check-task"

    mock_model = MagicMock(spec=clearml.OutputModel)

    model_info = WeightsFileHandler.ModelInfo(
        model=mock_model,
        upload_filename="test.pt",
        local_model_path="",
        local_model_id="",
        framework=Framework.pytorch,
        task=mock_task,
    )

    mock_model_info = MagicMock(spec_set=model_info)

    # Simulate 4 calls to save model and 2 to remove (n_saved=2)
    filenames = [
        "best_model_5_val_acc=0.123.pt",
        "best_model_6_val_acc=0.234.pt",
        "best_model_7_val_acc=0.356.pt",
        "best_model_8_val_acc=0.456.pt",
    ]
    metadata_list = [
        {"basename": "best_model", "score_name": "val_acc", "priority": 0.123},
        {"basename": "best_model", "score_name": "val_acc", "priority": 0.234},
        {"basename": "best_model", "score_name": "val_acc", "priority": 0.345},
        {"basename": "best_model", "score_name": "val_acc", "priority": 0.456},
    ]
    dirname = "/tmp/test"

    _checkpoint_slots = defaultdict(list)

    n_saved = 2

    for i, (filename, metadata) in enumerate(zip(filenames, metadata_list)):
        mock_model_info.upload_filename = filename

        if i >= n_saved:
            # Remove
            filename_to_remove = filenames[i % n_saved]
            for slots in _checkpoint_slots.values():
                try:
                    slots[slots.index(filename_to_remove)] = None
                except ValueError:
                    pass
                else:
                    i = i % n_saved
                    break

        basename = metadata["basename"]
        checkpoint_key = (dirname, basename)

        context = ClearMLSaver._CallbacksContext(
            callback_type=WeightsFileHandler.CallbackType,
            slots=_checkpoint_slots[checkpoint_key],
            checkpoint_key=str(checkpoint_key),
            filename=filename,
            basename=basename,
            metadata=metadata,
        )

        output_model_info = context.pre_callback(str(WeightsFileHandler.CallbackType.save), mock_model_info)
        assert (
            hasattr(output_model_info, "upload_filename") and f"{basename}_{i}.pt" in output_model_info.upload_filename
        )
        assert hasattr(output_model_info, "local_model_id") and str(checkpoint_key) in output_model_info.local_model_id

        output_model_info = context.post_callback(str(WeightsFileHandler.CallbackType.save), mock_model_info)
        assert hasattr(output_model_info, "model") and hasattr(output_model_info.model, "name")
        assert hasattr(output_model_info, "model") and hasattr(output_model_info.model, "comment")
        assert isinstance(output_model_info.model.name, str) and filename in output_model_info.model.name
        assert (
            isinstance(output_model_info.model.comment, str)
            and metadata["basename"] in output_model_info.model.comment
            and metadata["score_name"] in output_model_info.model.comment
        )


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.net(x)


def _test_save_model_optimizer_lr_scheduler_with_state_dict(device, on_zero_rank=False):
    if idist.get_rank() == 0:
        clearml.Task.current_task = MagicMock(spec=clearml.Task)
        clearml.binding.frameworks.WeightsFileHandler.create_output_model = MagicMock()

    torch.manual_seed(23)

    model = DummyModel().to(device)

    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)

    def update_fn(engine, batch):
        x = torch.rand((4, 2)).to(device)
        optim.zero_grad()
        y = model(x)
        # Below code raises: RuntimeError: torch_xla/csrc/tensor_impl.cpp:144 : XLA tensors do not have storage
        # Probably related to https://github.com/pytorch/xla/issues/2576
        # loss = y.pow(2.0).sum()
        loss = y.sum()
        loss.backward()
        if idist.has_xla_support:
            import torch_xla.core.xla_model as xm

            xm.optimizer_step(optim, barrier=True)
        else:
            optim.step()
        lr_scheduler.step()

    engine = Engine(update_fn)

    to_save = {"model": model, "optimizer": optim, "lr_scheduler": lr_scheduler}

    with pytest.warns(UserWarning, match=r"ClearMLSaver created a temporary checkpoints directory"):
        clearml_saver = ClearMLSaver()

    if (not on_zero_rank) or (on_zero_rank and idist.get_rank() == 0):
        checkpoint = Checkpoint(to_save=to_save, save_handler=clearml_saver, n_saved=1)
        engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    engine.run([0], max_epochs=4)

    idist.barrier()

    saved_objects = sorted(os.listdir(clearml_saver.dirname))
    # saved object is ['PREFIX_checkpoint_3.pt', ]
    saved_checkpoint = clearml_saver.dirname / saved_objects[0]

    if idist.has_xla_support:
        device = "cpu"

    loaded_obj = torch.load(saved_checkpoint, map_location=device)
    for f in ["model", "optimizer", "lr_scheduler"]:
        assert f in loaded_obj
    loaded_model_state_dict = loaded_obj["model"]
    loaded_optimizer_state_dict = loaded_obj["optimizer"]
    loaded_lr_scheduler_state_dict = loaded_obj["lr_scheduler"]

    assert isinstance(loaded_model_state_dict, dict)
    assert isinstance(loaded_optimizer_state_dict, dict)
    assert isinstance(loaded_lr_scheduler_state_dict, dict)

    # Specifically move device to CPU first
    model_state_dict = model.cpu().state_dict()
    for key in model_state_dict.keys():
        assert key in loaded_model_state_dict
        model_value = model_state_dict[key]
        loaded_model_value = loaded_model_state_dict[key]
        assert (model_value.cpu().numpy() == loaded_model_value.cpu().numpy()).all()

    optim_state_dict = optim.state_dict()
    for key in optim_state_dict.keys():
        assert key in loaded_optimizer_state_dict
        optim_value = optim_state_dict[key]
        loaded_optim_value = loaded_optimizer_state_dict[key]
        if idist.get_rank() == 0:
            assert optim_value == loaded_optim_value

    lr_scheduler_state_dict = lr_scheduler.state_dict()
    for key in lr_scheduler_state_dict.keys():
        assert key in loaded_lr_scheduler_state_dict
        lr_scheduler_value = lr_scheduler_state_dict[key]
        loaded_lr_scheduler_value = loaded_lr_scheduler_state_dict[key]
        assert lr_scheduler_value == loaded_lr_scheduler_value


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device)
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device, on_zero_rank=True)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device)
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device, on_zero_rank=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla():
    device = idist.device()
    assert "xla" in device.type
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device)


def _test_save_model_optimizer_lr_scheduler_with_state_dict_xla_nprocs(index):
    device = idist.device()
    _test_save_model_optimizer_lr_scheduler_with_state_dict(device)

    import time

    # hack to have all proc properly sync:
    time.sleep(1)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Not on TPU device")
def test_distrib_single_device_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_save_model_optimizer_lr_scheduler_with_state_dict_xla_nprocs, args=(), nprocs=n)
