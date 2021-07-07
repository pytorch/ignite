from typing import Callable, List, Optional, Tuple, Union

import torch


class GramMatrix(torch.nn.Module):
    def __init__(self) -> None:
        super(GramMatrix, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)

        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class StyleLoss(torch.nn.Module):
    def __init__(self, loss: Callable = torch.nn.functional.mse_loss) -> None:
        super(StyleLoss, self).__init__()
        self.gram_matrix = GramMatrix()
        self.loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:
        G_input = self.gram_matrix(input)
        G_target = self.gram_matrix(target)

        return self.loss(G_input, G_target)


class ContentLoss(torch.nn.Module):
    def __init__(self, loss: Callable = torch.nn.functional.mse_loss) -> None:
        super(ContentLoss, self).__init__()
        self.loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:
        return self.loss(input, target)


class PerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        normalize: bool = False,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        loss: Callable = torch.nn.functional.mse_loss,
        content_layers: List = [4],
        style_layers: List = [1, 2, 3, 4, 5],
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super(PerceptualLoss, self).__init__()

        if model is None:
            try:
                from torchvision import models
            except ImportError:
                raise RuntimeError("This module requires torchvision to use the default model.")
            self.model = models.vgg19(pretrained=True).features.to(device).eval()
        else:
            self.model = model

        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("model must be a torch Module, got %s" % type(model))

        if not callable(loss):
            raise ValueError("loss must be a Callable, got %s" % type(loss))

        if normalize and (mean is None or std is None):
            raise ValueError("mean and std arguments must be provided if normalize is True.")

        for layer in self.model.children():
            if isinstance(layer, torch.nn.ReLU):
                layer = torch.nn.ReLU(inplace=False)
            layer = layer.to(device)

        self.normalize = normalize
        self.mean = mean
        self.std = std

        self.style_layers = style_layers
        self.content_layers = content_layers

        self.gram_matrix = GramMatrix().to(device)
        self.style_loss = StyleLoss(loss).to(device)
        self.content_loss = ContentLoss(loss).to(device)

        self._device = device

    def forward(self, input: torch.Tensor, style_target: torch.Tensor, content_target: torch.Tensor) -> Tuple:

        if input.device != self._device:
            input = input.to(self._device)

        if style_target != self._device:
            style_target = style_target.to(self._device)

        if content_target != self._device:
            content_target = content_target.to(self._device)

        if self.normalize:
            input = (input - self.mean) / self.std
            style_target = (style_target - self.mean) / self.std
            content_target = (content_target - self.mean) / self.std

        style_losses = []
        content_losses = []

        for i, layer in enumerate(self.model.children()):
            content_target = layer(content_target)
            style_target = layer(style_target)
            input = layer(input)

            if i in self.style_layers:
                style_loss = self.style_loss(input, style_target)
                style_losses.append(style_loss)

            if i in self.content_layers:
                content_loss = self.content_loss(input, content_target)
                content_losses.append(content_loss)

        return style_losses, content_losses
