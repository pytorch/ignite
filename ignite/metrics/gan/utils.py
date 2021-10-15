from typing import Callable, Optional, Union

import torch

from ignite.metrics.metric import Metric


class InceptionModel(torch.nn.Module):
    r"""Inception Model pre-trained on the ImageNet Dataset.

    Args:
        return_features: set it to `True` if you want the model to return features from the last pooling
            layer instead of prediction probabilities.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """

    def __init__(self, return_features: bool, device: Union[str, torch.device] = "cpu") -> None:
        try:
            from torchvision import models
        except ImportError:
            raise RuntimeError("This module requires torchvision to be installed.")
        super(InceptionModel, self).__init__()
        self._device = device
        self.model = models.inception_v3(pretrained=True).to(self._device)
        if return_features:
            self.model.fc = torch.nn.Identity()
        else:
            self.model.fc = torch.nn.Sequential(self.model.fc, torch.nn.Softmax(dim=1))
        self.model.eval()

    @torch.no_grad()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        if data.device != torch.device(self._device):
            data = data.to(self._device)
        return self.model(data)


class _BaseInceptionMetric(Metric):
    def __init__(
        self,
        num_features: Optional[int],
        feature_extractor: Optional[torch.nn.Module],
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        if num_features is None:
            raise ValueError("Argument num_features must be provided, if feature_extractor is specified.")

        if feature_extractor is None:
            feature_extractor = torch.nn.Identity()

        if num_features <= 0:
            raise ValueError(f"Argument num_features must be greater to zero, got: {num_features}")

        if not isinstance(feature_extractor, torch.nn.Module):
            raise TypeError(
                f"Argument feature_extractor must be of type torch.nn.Module, got {type(self._feature_extractor)}"
            )

        self._num_features = num_features
        self._feature_extractor = feature_extractor.to(device)

        super(_BaseInceptionMetric, self).__init__(output_transform=output_transform, device=device)

    def _check_feature_shapes(self, samples: torch.Tensor) -> None:

        if samples.dim() != 2:
            raise ValueError(f"feature_extractor output must be a tensor of dim 2, got: {samples.dim()}")

        if samples.shape[0] == 0:
            raise ValueError(f"Batch size should be greater than one, got: {samples.shape[0]}")

        if samples.shape[1] != self._num_features:
            raise ValueError(
                f"num_features returned by feature_extractor should be {self._num_features}, got: {samples.shape[1]}"
            )

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:

        inputs = inputs.detach()

        if inputs.device != torch.device(self._device):
            inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._feature_extractor(inputs).to(self._device, dtype=torch.float64)
        self._check_feature_shapes(outputs)

        return outputs
