import os
from importlib.util import find_spec
from typing import Optional, Union
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from packaging.version import Version
from pytest import approx
from torch.nn.functional import mse_loss
from torch.optim import SGD

import ignite.distributed as idist
from ignite.engine import (
    _check_arg,
    create_supervised_evaluator,
    create_supervised_trainer,
    Engine,
    Events,
    supervised_evaluation_step,
    supervised_evaluation_step_amp,
    supervised_training_step_tpu,
)
from ignite.metrics import MeanSquaredError

from tests.ignite import is_mps_available_and_functional


class DummyModel(torch.nn.Module):
    def __init__(self, output_as_list=False):
        super(DummyModel, self).__init__()
        self.output_as_list = output_as_list
        self.fc = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x, bias=None):
        if bias is None:
            bias = 0.0
        if self.output_as_list:
            return self.fc(x) + bias, self.fc(x) + bias

        return self.fc(x) + bias


def _default_create_supervised_trainer(
    gradient_accumulation_steps: int = 1,
    model_device: Optional[str] = None,
    trainer_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
    scaler: Union[bool, "torch.amp.GradScaler"] = False,
    with_model_transform: bool = False,
    with_model_fn: bool = False,
):
    if with_model_transform:

        def get_first_element(output):
            return output[0]

        model = DummyModel(output_as_list=True)
        model_transform = get_first_element
    else:
        model = DummyModel()
        model_transform = None

    if model_device:
        model.to(model_device)

    model.fc.weight.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)

    if trace:
        example_inputs = (torch.randn(1), torch.randn(1)) if with_model_fn else torch.randn(1)
        model = torch.jit.trace(model, example_inputs)

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
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_transform=model_transform if model_transform is not None else lambda x: x,
        model_fn=(
            (lambda model, x: model(x, torch.tensor([0.01], device=model_device)))
            if with_model_fn
            else (lambda model, x: model(x))
        ),
    )
    assert model.fc.weight.data[0, 0].item() == approx(0.0)
    return trainer, model


def _test_create_supervised_trainer(
    gradient_accumulation_steps: int = 1,
    model_device: Optional[str] = None,
    trainer_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
    scaler: Union[bool, "torch.amp.GradScaler"] = False,
    with_model_transform: bool = False,
    with_model_fn: bool = False,
):
    trainer, model = _default_create_supervised_trainer(
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_device=model_device,
        trainer_device=trainer_device,
        trace=trace,
        amp_mode=amp_mode,
        scaler=scaler,
        with_model_transform=with_model_transform,
        with_model_fn=with_model_fn,
    )

    x = torch.tensor([[0.01], [0.02], [0.03], [0.04], [0.05]])
    y = torch.tensor([[0.015], [0.025], [0.035], [0.045], [0.055]])
    if with_model_fn:
        y += 0.01
    data = [(_x, _y) for _x, _y in zip(x, y)]

    theta = [0.0]
    accumulation = [0.0]
    loss = [0.0]

    @trainer.on(Events.ITERATION_COMPLETED)
    def _():
        assert model.fc.weight.grad != 0
        _x, _y = trainer.state.batch
        _x, _y = _x.to(model_device), _y.to(model_device)
        bias = 0.01 if with_model_fn else 0.0
        accumulation[0] += 0.2 * _x.item() * (theta[0] * _x.item() - (_y.item() - bias))
        # value of loss should not be accumulated
        _y_pred = model(_x, torch.tensor([bias], device=model_device)) if with_model_fn else model(_x)
        if with_model_transform:
            _y_pred = _y_pred[0]

        loss[0] = mse_loss(_y_pred, _y).item()

    @trainer.on(Events.ITERATION_COMPLETED(every=gradient_accumulation_steps))
    def _():
        theta[0] -= accumulation[0] / gradient_accumulation_steps
        assert pytest.approx(model.fc.weight.data[0, 0].item(), abs=1.0e-5) == theta[0]
        assert pytest.approx(trainer.state.output[-1], abs=1e-5) == loss[0]
        accumulation[0] = loss[0] = 0.0

    if model_device == trainer_device or ((model_device == "cpu") ^ (trainer_device == "cpu")):
        state = trainer.run(data)

        if amp_mode == "amp":
            assert state.output[0].dtype is torch.half
            if scaler and isinstance(scaler, bool):
                assert hasattr(state, "scaler")
            else:
                assert not hasattr(state, "scaler")

    else:
        if Version(torch.__version__) >= Version("1.7.0"):
            # This is broken in 1.6.0 but will be probably fixed with 1.7.0
            with pytest.raises(RuntimeError, match=r"Expected all tensors to be on the same device"):
                trainer.run(data)


@pytest.mark.skipif(Version(torch.__version__) < Version("2.3.1"), reason="Skip if < 2.3.1")
def test_create_supervised_training_scalar_assignment():
    with mock.patch("ignite.engine._check_arg") as check_arg_mock:
        check_arg_mock.return_value = None, torch.amp.GradScaler(enabled=False)
        trainer, _ = _default_create_supervised_trainer(model_device="cpu", trainer_device="cpu", scaler=True)
        assert hasattr(trainer.state, "scaler")
        assert isinstance(trainer.state.scaler, torch.amp.GradScaler)


def _test_create_mocked_supervised_trainer(
    model_device: Optional[str] = None,
    trainer_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
    scaler: Union[bool, "torch.amp.GradScaler"] = False,
):
    with mock.patch("ignite.engine.supervised_training_step_amp") as training_step_amp_mock:
        with mock.patch("ignite.engine.supervised_training_step_apex") as training_step_apex_mock:
            with mock.patch("ignite.engine.supervised_training_step_tpu") as training_step_tpu_mock:
                with mock.patch("ignite.engine.supervised_training_step") as training_step_mock:
                    trainer, _ = _default_create_supervised_trainer(
                        model_device=model_device,
                        trainer_device=trainer_device,
                        trace=trace,
                        amp_mode=amp_mode,
                        scaler=scaler,
                    )

                    x = torch.tensor([[0.1], [0.2]])
                    y = torch.tensor([[0.3], [0.5]])
                    data = [(x, y)]

                    on_tpu = "xla" in trainer_device if trainer_device is not None else False
                    on_mps = "mps" in trainer_device if trainer_device is not None else False
                    mode, _ = _check_arg(on_tpu, on_mps, amp_mode, scaler)

                    if model_device == trainer_device or ((model_device == "cpu") ^ (trainer_device == "cpu")):
                        trainer.run(data)

                        if mode == "amp":
                            assert training_step_amp_mock.called
                        elif mode == "apex":
                            assert training_step_apex_mock.called
                        elif mode == "tpu":
                            assert training_step_tpu_mock.called
                        else:
                            assert training_step_mock.called


def _test_create_supervised_trainer_wrong_accumulation(
    model_device=None, trainer_device=None, amp_mode=None, trace=False
):
    with pytest.raises(ValueError, match="Gradient_accumulation_steps must be strictly positive."):
        _default_create_supervised_trainer(
            gradient_accumulation_steps=0,
            model_device=model_device,
            trainer_device=trainer_device,
            amp_mode=amp_mode,
            trace=trace,
        )


def _default_create_supervised_evaluator(
    model_device: Optional[str] = None,
    evaluator_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
    with_model_transform: bool = False,
    with_model_fn: bool = False,
):
    if with_model_transform:

        def get_first_element(output):
            return output[0]

        model = DummyModel(output_as_list=True)
        model_transform = get_first_element
    else:
        model = DummyModel()
        model_transform = None

    if model_device:
        model.to(model_device)

    model.fc.weight.data.zero_()

    if trace:
        example_inputs = (torch.randn(1), torch.randn(1)) if with_model_fn else torch.randn(1)
        model = torch.jit.trace(model, example_inputs)

    evaluator = create_supervised_evaluator(
        model,
        device=evaluator_device,
        amp_mode=amp_mode,
        model_transform=model_transform if model_transform is not None else lambda x: x,
        model_fn=(
            (lambda model, x: model(x, torch.tensor([0.01], device=model_device)))
            if with_model_fn
            else (lambda model, x: model(x))
        ),
    )

    assert model.fc.weight.data[0, 0].item() == approx(0.0)

    return model, evaluator


def _test_create_supervised_evaluator(
    model_device: Optional[str] = None,
    evaluator_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
    with_model_transform: bool = False,
    with_model_fn: bool = False,
):
    model, evaluator = _default_create_supervised_evaluator(
        model_device=model_device,
        evaluator_device=evaluator_device,
        trace=trace,
        amp_mode=amp_mode,
        with_model_transform=with_model_transform,
        with_model_fn=with_model_fn,
    )
    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    if with_model_fn:
        y += 0.01
    data = [(x, y)]

    if model_device == evaluator_device or ((model_device == "cpu") ^ (evaluator_device == "cpu")):
        state = evaluator.run(data)

        y_pred, y = state.output
        if with_model_fn:
            y_pred -= 0.01
            y -= 0.01
        assert y_pred[0, 0].item() == approx(0.0)
        assert y_pred[1, 0].item() == approx(0.0)
        assert y[0, 0].item() == approx(3.0)
        assert y[1, 0].item() == approx(5.0)

        assert model.fc.weight.data[0, 0].item() == approx(0.0)

    else:
        if Version(torch.__version__) >= Version("1.7.0"):
            # This is broken in 1.6.0 but will be probably fixed with 1.7.0
            err_msg_1 = "Expected all tensors to be on the same device"
            err_msg_2 = "Placeholder storage has not been allocated on MPS device"
            err_msg_3 = "Tensor for argument weight is on cpu but expected on mps"
            with pytest.raises(RuntimeError, match=f"({err_msg_1}|{err_msg_2}|{err_msg_3})"):
                evaluator.run(data)


def _test_mocked_supervised_evaluator(
    model_device: Optional[str] = None,
    evaluator_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
):
    with mock.patch("ignite.engine.supervised_evaluation_step") as evaluation_step:
        with mock.patch("ignite.engine.supervised_evaluation_step_amp") as evaluation_step_amp:
            _, evaluator = _default_create_supervised_evaluator(
                model_device=model_device, evaluator_device=evaluator_device, trace=trace, amp_mode=amp_mode
            )

            x = torch.tensor([[1.0], [2.0]])
            y = torch.tensor([[3.0], [5.0]])
            data = [(x, y)]

            if model_device == evaluator_device or ((model_device == "cpu") ^ (evaluator_device == "cpu")):
                evaluator.run(data)

                if amp_mode == "amp":
                    assert evaluation_step_amp.called
                    assert not evaluation_step.called
                else:
                    assert evaluation_step.called
                    assert not evaluation_step_amp.called


def _test_create_evaluation_step_amp(
    autocast_mock,
    model_device: Optional[str] = None,
    evaluator_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
):
    output_transform_mock = MagicMock()
    model = DummyModel()

    if model_device:
        model.to(model_device)

    model.fc.weight.data.zero_()

    if trace:
        example_input = torch.randn(1, 1)
        model = torch.jit.trace(model, example_input)

    device_type = evaluator_device.type if isinstance(evaluator_device, torch.device) else evaluator_device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False
    mode, _ = _check_arg(on_tpu, on_mps, amp_mode, None)

    evaluate_step = supervised_evaluation_step_amp(model, evaluator_device, output_transform=output_transform_mock)

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]
    evaluator = Engine(evaluate_step)

    evaluator.run(data)
    assert autocast_mock.called
    assert output_transform_mock.called


def _test_create_evaluation_step(
    mock_torch_cuda_amp_module,
    model_device: Optional[str] = None,
    evaluator_device: Optional[str] = None,
    trace: bool = False,
    amp_mode: str = None,
):
    output_transform_mock = MagicMock()
    model = DummyModel()

    if model_device:
        model.to(model_device)

    model.fc.weight.data.zero_()

    if trace:
        example_input = torch.randn(1, 1)
        model = torch.jit.trace(model, example_input)

    device_type = evaluator_device.type if isinstance(evaluator_device, torch.device) else evaluator_device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False
    mode, _ = _check_arg(on_tpu, on_mps, amp_mode, None)

    evaluate_step = supervised_evaluation_step(model, evaluator_device, output_transform=output_transform_mock)

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]
    evaluator = Engine(evaluate_step)

    evaluator.run(data)
    assert not mock_torch_cuda_amp_module.called
    assert output_transform_mock.called


@pytest.mark.parametrize("trainer_device", [None, "cpu"])
@pytest.mark.parametrize("trace", [False, True])
def test_create_supervised_trainer(trainer_device, trace):
    _test_create_supervised_trainer_wrong_accumulation(trainer_device=trainer_device, trace=trace)
    _test_create_supervised_trainer(gradient_accumulation_steps=1, trainer_device=trainer_device, trace=trace)
    _test_create_supervised_trainer(gradient_accumulation_steps=3, trainer_device=trainer_device, trace=trace)
    _test_create_supervised_trainer(with_model_transform=True, trainer_device=trainer_device, trace=trace)
    _test_create_supervised_trainer(with_model_fn=True, trainer_device=trainer_device, trace=trace)
    _test_create_mocked_supervised_trainer(trainer_device=trainer_device, trace=trace)


@pytest.mark.skipif(find_spec("apex"), reason="Skip if APEX")
def test_create_supervised_trainer_apex_error():
    with pytest.raises(
        ModuleNotFoundError, match="Please install apex from https://github.com/nvidia/apex to use amp_mode='apex'."
    ):
        _test_create_supervised_trainer_wrong_accumulation(trainer_device="cpu", amp_mode="apex")
    with pytest.raises(
        ModuleNotFoundError, match="Please install apex from https://github.com/nvidia/apex to use amp_mode='apex'."
    ):
        _test_create_supervised_trainer(amp_mode="apex")


@pytest.fixture
def mock_torch_cuda_amp_module():
    with patch.dict(
        "sys.modules",
        {"torch.amp": None, "torch.cuda.amp": None, "torch.amp.autocast_mode": None},
    ):
        yield torch


def test_create_supervised_trainer_amp_error(mock_torch_cuda_amp_module):
    with pytest.raises(ImportError, match="Please install torch>=1.12.0 to use amp_mode='amp'."):
        _test_create_supervised_trainer_wrong_accumulation(trainer_device="cpu", amp_mode="amp")
    with pytest.raises(ImportError, match="Please install torch>=1.12.0 to use amp_mode='amp'."):
        _test_create_supervised_trainer(amp_mode="amp")
    with pytest.raises(ImportError, match="Please install torch>=2.3.1 to use scaler argument."):
        _test_create_supervised_trainer(amp_mode="amp", scaler=True)


@pytest.mark.skipif(Version(torch.__version__) < Version("2.3.1"), reason="Skip if < 2.3.1")
def test_create_supervised_trainer_scaler_not_amp():
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    with pytest.raises(ValueError, match=f"scaler argument is {scaler}, but amp_mode is None."):
        _test_create_supervised_trainer(amp_mode=None, scaler=scaler)
    with pytest.raises(ValueError, match="scaler argument is True, but amp_mode is None."):
        _test_create_supervised_trainer(amp_mode=None, scaler=True)
    with pytest.raises(ValueError, match="scaler argument is True, but amp_mode is apex."):
        _test_create_supervised_trainer(amp_mode="apex", scaler=True)
    with pytest.raises(ValueError, match=f"scaler argument is {scaler}, but amp_mode is apex."):
        _test_create_supervised_trainer(amp_mode="apex", scaler=scaler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer_wrong_accumulation(model_device=model_device, trainer_device=trainer_device)
    _test_create_supervised_trainer(
        gradient_accumulation_steps=1, model_device=model_device, trainer_device=trainer_device
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=3, model_device=model_device, trainer_device=trainer_device
    )
    _test_create_mocked_supervised_trainer(model_device=model_device, trainer_device=trainer_device)


@pytest.mark.skipif(not is_mps_available_and_functional(), reason="Skip if no MPS")
def test_create_supervised_trainer_on_mps():
    model_device = trainer_device = "mps"
    _test_create_supervised_trainer_wrong_accumulation(model_device=model_device, trainer_device=trainer_device)
    _test_create_supervised_trainer(
        gradient_accumulation_steps=1, model_device=model_device, trainer_device=trainer_device
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=3, model_device=model_device, trainer_device=trainer_device
    )
    _test_create_mocked_supervised_trainer(model_device=model_device, trainer_device=trainer_device)


@pytest.mark.skipif(Version(torch.__version__) < Version("1.12.0"), reason="Skip if < 1.12.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda_amp():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer_wrong_accumulation(
        model_device=model_device, trainer_device=trainer_device, amp_mode="amp"
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=1, model_device=model_device, trainer_device=trainer_device, amp_mode="amp"
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=3, model_device=model_device, trainer_device=trainer_device, amp_mode="amp"
    )
    _test_create_mocked_supervised_trainer(model_device=model_device, trainer_device=trainer_device, amp_mode="amp")


@pytest.mark.skipif(Version(torch.__version__) < Version("1.12.0"), reason="Skip if < 1.12.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda_amp_scaler():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer_wrong_accumulation(
        model_device=model_device, trainer_device=trainer_device, amp_mode="amp"
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=1,
        model_device=model_device,
        trainer_device=trainer_device,
        amp_mode="amp",
        scaler=True,
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=3,
        model_device=model_device,
        trainer_device=trainer_device,
        amp_mode="amp",
        scaler=True,
    )
    _test_create_mocked_supervised_trainer(
        model_device=model_device, trainer_device=trainer_device, amp_mode="amp", scaler=True
    )
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    _test_create_supervised_trainer(
        gradient_accumulation_steps=1,
        model_device=model_device,
        trainer_device=trainer_device,
        amp_mode="amp",
        scaler=scaler,
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=3,
        model_device=model_device,
        trainer_device=trainer_device,
        amp_mode="amp",
        scaler=scaler,
    )
    _test_create_mocked_supervised_trainer(
        model_device=model_device, trainer_device=trainer_device, amp_mode="amp", scaler=scaler
    )


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
# @pytest.mark.skipif(not find_spec("apex"), reason="Skip if no APEX")
@pytest.mark.skip(reason="Temporarily disabled, as it fails because of an issue from apex side")
def test_create_supervised_trainer_on_cuda_apex():
    model_device = trainer_device = "cuda"
    _test_create_supervised_trainer_wrong_accumulation(
        model_device=model_device, trainer_device=trainer_device, amp_mode="apex"
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=1, model_device=model_device, trainer_device=trainer_device, amp_mode="apex"
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=3, model_device=model_device, trainer_device=trainer_device, amp_mode="apex"
    )
    _test_create_mocked_supervised_trainer(model_device=model_device, trainer_device=trainer_device, amp_mode="apex")


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
    _test_create_supervised_trainer_wrong_accumulation(model_device=model_device, trainer_device=trainer_device)
    _test_create_supervised_trainer(
        gradient_accumulation_steps=1, model_device=model_device, trainer_device=trainer_device
    )
    _test_create_supervised_trainer(
        gradient_accumulation_steps=3, model_device=model_device, trainer_device=trainer_device
    )
    _test_create_mocked_supervised_trainer(model_device=model_device, trainer_device=trainer_device)


@pytest.mark.tpu
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_create_supervised_trainer_on_tpu_amp():
    model_device = trainer_device = "xla"
    with pytest.raises(ValueError, match="amp_mode cannot be used with xla device."):
        _test_create_supervised_trainer(model_device=model_device, trainer_device=trainer_device, amp_mode="amp")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda_with_model_on_cpu():
    _test_create_supervised_trainer_wrong_accumulation(trainer_device="cuda")
    _test_create_supervised_trainer(gradient_accumulation_steps=1, trainer_device="cuda")
    _test_create_supervised_trainer(gradient_accumulation_steps=3, trainer_device="cuda")
    _test_create_mocked_supervised_trainer(trainer_device="cuda")


def test_create_supervised_evaluator():
    _test_create_supervised_evaluator()
    _test_create_supervised_evaluator(with_model_transform=True)
    _test_create_supervised_evaluator(with_model_fn=True)
    _test_mocked_supervised_evaluator()

    # older versions didn't have the autocast method so we skip the test for older builds
    if Version(torch.__version__) >= Version("1.12.0"):
        with mock.patch("torch.amp.autocast") as mock_torch_cuda_amp_module:
            _test_create_evaluation_step_amp(mock_torch_cuda_amp_module)


def test_create_supervised_evaluator_on_cpu():
    _test_create_supervised_evaluator(evaluator_device="cpu")
    _test_mocked_supervised_evaluator(evaluator_device="cpu")

    # older versions didn't have the autocast method so we skip the test for older builds
    if Version(torch.__version__) >= Version("1.12.0"):
        with mock.patch("torch.amp.autocast") as mock_torch_cuda_amp_module:
            _test_create_evaluation_step(mock_torch_cuda_amp_module, evaluator_device="cpu")
            _test_create_evaluation_step_amp(mock_torch_cuda_amp_module, evaluator_device="cpu")


def test_create_supervised_evaluator_traced_on_cpu():
    _test_create_supervised_evaluator(evaluator_device="cpu", trace=True)
    _test_mocked_supervised_evaluator(evaluator_device="cpu", trace=True)

    # older versions didn't have the autocast method so we skip the test for older builds
    if Version(torch.__version__) >= Version("1.12.0"):
        with mock.patch("torch.amp.autocast") as mock_torch_cuda_amp_module:
            _test_create_evaluation_step(mock_torch_cuda_amp_module, evaluator_device="cpu", trace=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_evaluator_on_cuda():
    model_device = evaluator_device = "cuda"
    _test_create_supervised_evaluator(model_device=model_device, evaluator_device=evaluator_device)
    _test_mocked_supervised_evaluator(model_device=model_device, evaluator_device=evaluator_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_evaluator_on_cuda_with_model_on_cpu():
    _test_create_supervised_evaluator(evaluator_device="cuda")
    _test_mocked_supervised_evaluator(evaluator_device="cuda")


@pytest.mark.skipif(not is_mps_available_and_functional(), reason="Skip if no MPS")
def test_create_supervised_evaluator_on_mps():
    model_device = evaluator_device = "mps"
    _test_create_supervised_evaluator(model_device=model_device, evaluator_device=evaluator_device)
    _test_mocked_supervised_evaluator(model_device=model_device, evaluator_device=evaluator_device)


@pytest.mark.skipif(not is_mps_available_and_functional(), reason="Skip if no MPS")
def test_create_supervised_evaluator_on_mps_with_model_on_cpu():
    _test_create_supervised_evaluator(evaluator_device="mps")
    _test_mocked_supervised_evaluator(evaluator_device="mps")


@pytest.mark.skipif(Version(torch.__version__) < Version("1.12.0"), reason="Skip if < 1.12.0")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_evaluator_on_cuda_amp():
    model_device = evaluator_device = "cuda"
    _test_create_supervised_evaluator(model_device=model_device, evaluator_device=evaluator_device, amp_mode="amp")
    _test_mocked_supervised_evaluator(model_device=model_device, evaluator_device=evaluator_device, amp_mode="amp")


def test_create_supervised_evaluator_amp_error(mock_torch_cuda_amp_module):
    with pytest.raises(ImportError, match="Please install torch>=1.12.0 to use amp_mode='amp'."):
        _test_create_supervised_evaluator(amp_mode="amp")


def test_create_supervised_evaluator_with_metrics():
    model = DummyModel()
    model.fc.weight.data.zero_()

    evaluator = create_supervised_evaluator(model, metrics={"mse": MeanSquaredError()})

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [4.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    assert state.metrics["mse"] == 12.5
