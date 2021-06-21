from unittest.mock import patch

import pytest
import torch
import torchvision

from ignite.metrics.gan.utils import InceptionModel


def test_inception_extractor_wrong_inputs():
    with pytest.raises(ValueError, match=r"Inputs should be a tensor of dim 4"):
        InceptionModel(return_features=True)(torch.rand(2))
    with pytest.raises(ValueError, match=r"Inputs should be a tensor with 3 channels"):
        InceptionModel(return_features=True)(torch.rand(2, 2, 2, 0))


@pytest.fixture()
def mock_no_torchvision():
    with patch.dict("sys.modules", {"torchvision": None}):
        yield torchvision


def test_no_torchvision(mock_no_torchvision):
    with pytest.raises(RuntimeError, match=r"This module requires torchvision to be installed."):
        InceptionModel(return_features=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_device_mismatch():
    images = torch.rand(10, 3, 299, 299)
    assert InceptionModel(return_features=False, device="cuda")(images).shape == torch.Size([10, 1000])
