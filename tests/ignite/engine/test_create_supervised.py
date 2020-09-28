import os
from distutils.version import LooseVersion
from typing import Optional

import pytest
import torch
from pytest import approx
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD

import ignite.distributed as idist
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import MeanSquaredError


def _test_create_supervised_trainer(
    model_device: Optional[str] = None, trainer_device: Optional[str] = None, trace: bool = False
):
    model = Linear(1, 1)

    if model_device:
        model.to(model_device)

    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)

    if trace:
        example_input = torch.randn(1, 1)
        model = torch.jit.trace(model, example_input)

    trainer = create_supervised_trainer(model, optimizer, mse_loss, device=trainer_device)

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    if model_device == trainer_device or ((model_device == "cpu") ^ (trainer_device == "cpu")):
        state = trainer.run(data)

        assert state.output == approx(17.0)
        assert model.weight.data[0, 0].item() == approx(1.3)
        assert model.bias.item() == approx(0.8)
    else:
        if LooseVersion(torch.__version__) >= LooseVersion("1.7.0"):
            # This is broken in 1.6.0 but will be probably fixed with 1.7.0
            with pytest.raises(RuntimeError, match=r"is on CPU, but expected them to be on GPU"):
                trainer.run(data)


def _test_create_supervised_evaluator(
    model_device: Optional[str] = None, evaluator_device: Optional[str] = None, trace: bool = False
):
    model = Linear(1, 1)

    if model_device:
        model.to(model_device)

    model.weight.data.zero_()
    model.bias.data.zero_()

    if trace:
        example_input = torch.randn(1, 1)
        model = torch.jit.trace(model, example_input)

    evaluator = create_supervised_evaluator(model, device=evaluator_device)

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    if model_device == evaluator_device or ((model_device == "cpu") ^ (evaluator_device == "cpu")):
        state = evaluator.run(data)

        y_pred, y = state.output

        assert y_pred[0, 0].item() == approx(0.0)
        assert y_pred[1, 0].item() == approx(0.0)
        assert y[0, 0].item() == approx(3.0)
        assert y[1, 0].item() == approx(5.0)

        assert model.weight.data[0, 0].item() == approx(0.0)
        assert model.bias.item() == approx(0.0)

    else:
        if LooseVersion(torch.__version__) >= LooseVersion("1.7.0"):
            # This is broken in 1.6.0 but will be probably fixed with 1.7.0
            with pytest.raises(RuntimeError, match=r"is on CPU, but expected them to be on GPU"):
                evaluator.run(data)


def test_create_supervised_trainer():
    _test_create_supervised_trainer()


def test_create_supervised_trainer_with_cpu():
    _test_create_supervised_trainer(trainer_device="cpu")


def test_create_supervised_trainer_traced_with_cpu():
    _test_create_supervised_trainer(trainer_device="cpu", trace=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device)


@pytest.mark.skipif(idist.has_xla_support, reason="Skip if has PyTorch XLA package")
def test_create_supervised_trainer_on_tpu_no_xla():
    model_device = "cpu"
    trainer_device = "xla"
    with pytest.raises(RuntimeError, match=r"In order to run on TPU, please install PyTorch XLA"):
        _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_create_supervised_trainer_on_tpu():
    model_device = trainer_device = "xla"
    _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda_with_model_on_cpu():
    _test_create_supervised_trainer(trainer_device="cuda")


def test_create_supervised_evaluator():
    _test_create_supervised_evaluator()


def test_create_supervised_evaluator_on_cpu():
    _test_create_supervised_evaluator(evaluator_device="cpu")


def test_create_supervised_evaluator_traced_on_cpu():
    _test_create_supervised_evaluator(evaluator_device="cpu", trace=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_evaluator_on_cuda():
    model_device = evaluator_device = "cuda"
    _test_create_supervised_evaluator(model_device=model_device, evaluator_device=evaluator_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_evaluator_on_cuda_with_model_on_cpu():
    _test_create_supervised_evaluator(evaluator_device="cuda")


def test_create_supervised_evaluator_with_metrics():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model, metrics={"mse": MeanSquaredError()})

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [4.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    assert state.metrics["mse"] == 12.5
