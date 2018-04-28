import pytest
import torch
from torch.autograd import Variable
from ignite._utils import convert_tensor, to_onehot


def test_convert_tensor():
    tensor = torch.Tensor([1])
    assert not tensor.requires_grad
    tensor = convert_tensor(tensor, requires_grad=True)
    assert tensor.requires_grad

    # 'convert_tensor' will not catch exceptions raised by pytorch.
    leaf = torch.Tensor([1])
    leaf.requires_grad_(True)
    not_leaf = leaf + 2
    with pytest.raises(RuntimeError):
        convert_tensor(not_leaf, requires_grad=False)


def test_to_onehot():
    indices = torch.LongTensor([0, 1, 2, 3])
    actual = to_onehot(indices, 4)
    expected = torch.eye(4)
    assert actual.equal(expected)
