from typing import Callable, Optional, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class PerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        function: torch.nn.Module = torch.nn.Identity(),
        normalize: bool = False,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        loss: Callable = torch.nn.functional.mse_loss,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super(PerceptualLoss, self).__init__()

        if not isinstance(function, torch.nn.Module):
            raise TypeError(f"Argument function must be of type torch.nn.Module, got {type(function)}")

        self.function = function
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.loss = loss
        self._device = device

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:

        input = input.to(self._device)
        target = target.to(self._device)

        if self.normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        input = self.function(input)
        target = self.function(target)

        return self.loss(input, target)


class GramMatrix(torch.nn.Module):
    def __init__(self) -> None:
        super(GramMatrix, self).__init__()

    def forward(self, input: torch.Tensor) -> float:
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


def get_style_loss_object(device: Union[str, torch.device] = torch.device("cpu")) -> PerceptualLoss:
    return PerceptualLoss(function=GramMatrix(), device=device)


def get_features_loss_object(device: Union[str, torch.device] = torch.device("cpu")) -> PerceptualLoss:

    return PerceptualLoss(
        normalize=True,
        mean=torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1),
        std=torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1),
        device=device,
    )


class PPL(Metric):
    def __init__(
        self,
        loss_model: torch.nn.Module,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self._loss_model = loss_model
        super(PPL, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._total_loss = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(PPL, self).reset()

    @reinit__is_reduced
    def update(self, output: torch.Tensor) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        self._check_wrong_inputs(y_pred, y)

        self._total_loss += self._loss_model(y_pred, y).to(self._device)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_num_examples", "_total_loss")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("PPL must have at least one example before it can be computed.")

        return self._total_loss.item() / self._num_examples
