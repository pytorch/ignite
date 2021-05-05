from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.fixture()
def norm_mock():
    def norm(x):
        return np.linalg.norm(x)

    norm_mock = Mock(side_effect=norm, spec=norm)
    norm_mock.configure_mock(**{"__name__": "norm"})
    norm_mock.reset_mock()
    return norm_mock


@pytest.fixture()
def dummy_model_factory():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc1 = torch.nn.Linear(10, 10)
            self.fc2 = torch.nn.Linear(12, 12)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.fill_(1.0)
            self.fc2.bias.data.fill_(1.0)

    def get_dummy_model(with_grads=True, with_frozen_layer=False):
        model = DummyModel()
        if with_grads:
            model.fc2.weight.grad = torch.zeros_like(model.fc2.weight)
            model.fc2.bias.grad = torch.zeros_like(model.fc2.bias)

            if not with_frozen_layer:
                model.fc1.weight.grad = torch.zeros_like(model.fc1.weight)
                model.fc1.bias.grad = torch.zeros_like(model.fc1.bias)

        if with_frozen_layer:
            for param in model.fc1.parameters():
                param.requires_grad = False
        return model

    return get_dummy_model
