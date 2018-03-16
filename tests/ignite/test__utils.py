import pytest
import torch
from torch.autograd import Variable
from ignite._utils import to_variable, to_tensor


def test_to_variable():
    x = torch.Tensor([0.0])
    var_x = to_variable(x)
    assert isinstance(var_x, Variable)

    x = (torch.Tensor([0.0]), torch.Tensor([0.0]))
    var_x = to_variable(x)
    assert isinstance(var_x, list) and isinstance(var_x[0], Variable) and isinstance(var_x[1], Variable)

    x = {'a': torch.Tensor([0.0]), 'b': torch.Tensor([0.0])}
    var_x = to_variable(x)
    assert isinstance(var_x, dict) and isinstance(var_x['a'], Variable) and isinstance(var_x['b'], Variable)

    with pytest.raises(TypeError):
        to_variable(12345)


def test_to_tensor():
    var_x = Variable(torch.Tensor([0.0]))
    x = to_tensor(var_x)
    assert torch.is_tensor(x)

    var_x = (Variable(torch.Tensor([0.0])), Variable(torch.Tensor([0.0])))
    x = to_tensor(var_x)
    assert isinstance(x, list) and torch.is_tensor(x[0]) and torch.is_tensor(x[1])

    var_x = {'a': Variable(torch.Tensor([0.0])), 'b': Variable(torch.Tensor([0.0]))}
    x = to_tensor(var_x)
    assert isinstance(x, dict) and torch.is_tensor(x['a']) and torch.is_tensor(x['b'])

    with pytest.raises(TypeError):
        to_tensor(12345)
