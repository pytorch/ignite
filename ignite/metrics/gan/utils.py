from typing import Callable, Optional, Tuple, Union

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
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        if num_features is not None and num_features <= 0:
            raise ValueError(f"Argument num_features must be greater to zero, got: {num_features}")
        super(_BaseInceptionMetric, self).__init__(output_transform=output_transform, device=device)

    def _check_input(
        self,
        num_features: Optional[int],
        feature_extractor: Optional[torch.nn.Module],
        device: Union[str, torch.device],
    ) -> Tuple[int, torch.nn.Module]:
        if num_features is None and feature_extractor is None:
            return self._default_channels, self._default_eval_model(self._default_args)
        elif num_features is None:
            raise ValueError("Argument num_features should be defined, if feature_extractor is provided")
        elif feature_extractor is None:
            return num_features, torch.nn.Identity()
        elif not isinstance(feature_extractor, torch.nn.Module):
            raise TypeError(
                f"Argument feature_extractor must be of type torch.nn.Module, got {type(feature_extractor)}"
            )
        else:
            return num_features, feature_extractor.to(device)

    def _check_feature_input(self, samples: torch.Tensor, num_features: int) -> None:
        if samples.dim() != 2:
            raise ValueError(f"eval_model output must be a tensor of dim 2, got: {samples.dim()}")
        if samples.shape[0] == 0:
            raise ValueError(f"Batch size should be greater than one, got: {samples.shape[0]}")
        if samples.shape[1] != num_features:
            raise ValueError(f"num_features returned by eval_model should be {num_features}, got: {samples.shape[1]}")
