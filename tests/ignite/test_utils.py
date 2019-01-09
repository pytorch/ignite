import pytest
import torch
from ignite.utils import convert_tensor, to_onehot


def test_convert_tensor():
    x = torch.Tensor([0.0])
    tensor = convert_tensor(x)
    assert torch.is_tensor(tensor)

    x = torch.Tensor([0.0])
    tensor = convert_tensor(x, device='cpu', non_blocking=True)
    assert torch.is_tensor(tensor)

    x = torch.Tensor([0.0])
    tensor = convert_tensor(x, device='cpu', non_blocking=False)
    assert torch.is_tensor(tensor)

    x = (torch.Tensor([0.0]), torch.Tensor([0.0]))
    list_ = convert_tensor(x)
    assert isinstance(list_, list)
    assert torch.is_tensor(list_[0])
    assert torch.is_tensor(list_[1])

    x = {'a': torch.Tensor([0.0]), 'b': torch.Tensor([0.0])}
    dict_ = convert_tensor(x)
    assert isinstance(dict_, dict)
    assert torch.is_tensor(dict_['a'])
    assert torch.is_tensor(dict_['b'])

    assert convert_tensor('a') == 'a'

    with pytest.raises(TypeError):
        convert_tensor(12345)


def test_to_onehot():
    indices = torch.LongTensor([0, 1, 2, 3])
    actual = to_onehot(indices, 4)
    expected = torch.eye(4)
    assert actual.equal(expected)
