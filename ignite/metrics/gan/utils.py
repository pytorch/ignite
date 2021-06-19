import torch


class InceptionModel:
    def __init__(self, return_features) -> None:
        try:
            from torchvision import models
        except ImportError:
            raise RuntimeError("This module requires torchvision to be installed.")
        self.model = models.inception_v3(pretrained=True)
        if return_features:
            self.model.fc = torch.nn.Identity()
        self.model.eval()

    @torch.no_grad()
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        return self.model(data)
