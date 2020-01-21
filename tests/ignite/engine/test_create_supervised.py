import pytest
from pytest import approx

import torch
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import SGD

from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanSquaredError


def test_create_supervised_trainer():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    trainer = create_supervised_trainer(model, optimizer, mse_loss)

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert model.weight.data[0, 0].item() == approx(1.3)
    assert model.bias.item() == approx(0.8)


def test_create_supervised_trainer_with_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    trainer = create_supervised_trainer(model, optimizer, mse_loss, device='cpu')

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert model.weight.data[0, 0].item() == approx(1.3)
    assert model.bias.item() == approx(0.8)


def test_create_supervised_trainer_traced_with_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    example_input = torch.randn(1, 1)
    traced_model = torch.jit.trace(model, example_input)

    optimizer = SGD(traced_model.parameters(), 0.1)

    trainer = create_supervised_trainer(traced_model, optimizer, mse_loss, device='cpu')

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    assert traced_model.weight.data[0, 0].item() == approx(0.0)
    assert traced_model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert traced_model.weight.data[0, 0].item() == approx(1.3)
    assert traced_model.bias.item() == approx(0.8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_trainer_on_cuda():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    trainer = create_supervised_trainer(model, optimizer, mse_loss, device='cuda')

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)

    state = trainer.run(data)

    assert state.output == approx(17.0)
    assert model.weight.data[0, 0].item() == approx(1.3)
    assert model.bias.item() == approx(0.8)


def test_create_supervised():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model)

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)


def test_create_supervised_on_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model, device='cpu')

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)


def test_create_supervised_evaluator_traced_on_cpu():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    example_input = torch.randn(1, 1)
    traced_model = torch.jit.trace(model, example_input)

    evaluator = create_supervised_evaluator(traced_model, device='cpu')

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert traced_model.weight.data[0, 0].item() == approx(0.0)
    assert traced_model.bias.item() == approx(0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_on_cuda():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model, device='cuda')

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [5.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    y_pred, y = state.output

    assert y_pred[0, 0].item() == approx(0.0)
    assert y_pred[1, 0].item() == approx(0.0)
    assert y[0, 0].item() == approx(3.0)
    assert y[1, 0].item() == approx(5.0)

    assert model.weight.data[0, 0].item() == approx(0.0)
    assert model.bias.item() == approx(0.0)


def test_create_supervised_with_metrics():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()

    evaluator = create_supervised_evaluator(model, metrics={'mse': MeanSquaredError()})

    x = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[3.0], [4.0]])
    data = [(x, y)]

    state = evaluator.run(data)
    assert state.metrics['mse'] == 12.5
