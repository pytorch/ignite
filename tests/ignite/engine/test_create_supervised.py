import os
from distutils.version import LooseVersion
from importlib.util import find_spec
from typing import Optional, Union
from unittest.mock import patch

import pytest
import torch
from pytest import approx
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD

import ignite.distributed as idist
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, supervised_training_step_tpu
from ignite.metrics import MeanSquaredError


def _test_create_supervised_trainer(
    model_device: Optional[str] = None,
    trainer_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
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

    if amp_mode == "apex" and model_device == trainer_device == "cuda":
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    trainer = create_supervised_trainer(
        model,
        optimizer,
        mse_loss,
        device=trainer_device,
        output_transform=lambda x, y, y_pred, loss: (y_pred, loss.item()),
        amp_mode=amp_mode,
        scaler=scaler,
    )

    x = torch.tensor([[0.1], [0.2]])
    y = torch.tensor([[0.3], [0.5]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    if model_device == trainer_device or ((model_device == "cpu") ^ (trainer_device == "cpu")):
        state = trainer.run(data)

        assert state.output[-1] == approx(0.17), state.output[-1]
        assert round(model.weight.data[0, 0].item(), 3) == approx(0.013), model.weight.item()
        assert round(model.bias.item(), 3) == approx(0.08), model.bias.item()

        if amp_mode == "amp":
            assert state.output[0].dtype is torch.half
            if scaler and isinstance(scaler, bool):
                assert hasattr(state, "scaler")
            else:
                assert not hasattr(state, "scaler")
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


@pytest.mark.skipif(find_spec("apex"), reason="Skip if APEX")
def test_create_supervised_trainer_apex_error():
    with pytest.raises(
        ModuleNotFoundError, match="Please install apex from https://github.com/nvidia/apex to use amp_mode='apex'."
    ):
        _test_create_supervised_trainer(amp_mode="apex")


@pytest.fixture
def mock_torch_cuda_amp_module():
    with patch.dict(
        "sys.modules",
        {"torch.cuda.amp": None, "torch.cuda.amp.grad_scaler": None, "torch.cuda.amp.autocast_mode": None},
    ):
        yield torch


def test_create_supervised_trainer_amp_error(mock_torch_cuda_amp_module):
    with pytest.raises(ImportError, match="Please install torch>=1.6.0 to use amp_mode='amp'."):
        _test_create_supervised_trainer(amp_mode="amp")
    with pytest.raises(ImportError, match="Please install torch>=1.6.0 to use scaler argument."):
        _test_create_supervised_trainer(amp_mode="amp", scaler=True)


@pytest.mark.skipif(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Skip if < 1.6.0")
def test_create_supervised_trainer_scaler_not_amp():
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    with pytest.warns(UserWarning, match=f"scaler argument is {scaler}, but amp_mode is None."):
        _test_create_supervised_trainer(amp_mode=None, scaler=scaler)
    with pytest.warns(UserWarning, match="scaler argument is True, but amp_mode is None."):
        _test_create_supervised_trainer(amp_mode=None, scaler=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device)


@pytest.mark.skipif(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Skip if < 1.6.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda_amp():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device, amp_mode="amp")


@pytest.mark.skipif(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Skip if < 1.6.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda_amp_scaler():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer(
        model_device=model_device, trainer_device=trainer_device, amp_mode="amp", scaler=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    _test_create_supervised_trainer(
        model_device=model_device, trainer_device=trainer_device, amp_mode="amp", scaler=scaler
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
@pytest.mark.skipif(not find_spec("apex"), reason="Skip if no APEX")
def test_create_supervised_trainer_on_cuda_apex():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device, amp_mode="apex")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
@pytest.mark.skipif(not find_spec("apex"), reason="Skip if no APEX")
@pytest.mark.skipif(LooseVersion(torch.__version__) < LooseVersion("1.6.0"), reason="Skip if < 1.6.0")
def test_create_supervised_trainer_on_cuda_apex_scaler():
    model_device = trainer_device = "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    with pytest.warns(UserWarning, match="scaler argument is True, but amp_mode is apex."):
        _test_create_supervised_trainer(
            model_device=model_device, trainer_device=trainer_device, amp_mode="apex", scaler=True
        )
    with pytest.warns(UserWarning, match=f"scaler argument is {scaler}, but amp_mode is apex."):
        _test_create_supervised_trainer(
            model_device=model_device, trainer_device=trainer_device, amp_mode="apex", scaler=scaler
        )


@pytest.mark.skipif(idist.has_xla_support, reason="Skip if has PyTorch XLA package")
def test_supervised_training_step_tpu_no_xla():
    with pytest.raises(ModuleNotFoundError, match="torch_xla cannot be imported, please install PyTorch XLA."):
        supervised_training_step_tpu(model=None, optimizer=None, loss_fn=None)


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


@pytest.mark.tpu
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_create_supervised_trainer_on_tpu_amp():
    model_device = trainer_device = "xla"
    with pytest.raises(ValueError, match="amp_mode cannot be used with xla device."):
        _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device, amp_mode="amp")


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
