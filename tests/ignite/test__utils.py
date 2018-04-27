import pytest
import torch
from torch.autograd import Variable
from ignite._utils import to_tensor, to_onehot


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

    x = torch.ones(1, requires_grad=True)
    y = to_tensor(x + 2, requires_grad=False)
    z = y + 2
    assert not y.requires_grad
    assert not z.requires_grad


def test_to_onehot():
    indices = torch.LongTensor([0, 1, 2, 3])
    actual = to_onehot(indices, 4)
    expected = torch.eye(4)
    assert actual.equal(expected)
