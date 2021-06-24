from typing import Callable, Optional, Union
from unittest.mock import patch

import pytest
import torch
import torchvision

from ignite.metrics.gan.utils import InceptionModel, _BaseInceptionMetric


class DummyInceptionMetric(_BaseInceptionMetric):
    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super(DummyInceptionMetric, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    def reset(self):
        pass

    def compute(self):
        pass

    def update(self, output):
        self._extract_features(output)


def test_dummy_metric():

    with pytest.raises(ValueError, match=r"Argument num_features must be greater to zero, got:"):
        DummyInceptionMetric(num_features=-1, feature_extractor=torch.nn.Identity()).update(torch.rand(2, 0))

    with pytest.raises(ValueError, match=r"feature_extractor output must be a tensor of dim 2, got: 1"):
        DummyInceptionMetric(num_features=1000, feature_extractor=torch.nn.Identity()).update(torch.rand(3))

    with pytest.raises(ValueError, match=r"Batch size should be greater than one, got: 0"):
        DummyInceptionMetric(num_features=1000, feature_extractor=torch.nn.Identity()).update(torch.rand(0, 0))

    with pytest.raises(ValueError, match=r"num_features returned by feature_extractor should be 1000, got: 0"):
        DummyInceptionMetric(num_features=1000, feature_extractor=torch.nn.Identity()).update(torch.rand(2, 0))

    with pytest.raises(ValueError, match=r"Argument num_features must be provided, if feature_extractor is specified."):
        DummyInceptionMetric(feature_extractor=torch.nn.Identity())


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
def test_device_mismatch_cuda():
    images = torch.rand(10, 3, 299, 299)
    assert InceptionModel(return_features=False, device="cuda")(images).shape == torch.Size([10, 1000])
