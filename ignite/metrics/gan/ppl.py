from typing import Callable, Iterable, Optional, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class PerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        model: Iterable,
        layer_weights: torch.Tensor,
        normalize: bool = False,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        loss: Callable = torch.nn.functional.mse_loss,
    ) -> None:
        super(PerceptualLoss, self).__init__()
        self.model = model
        for layer in model:
            layer.eval()
            layer.requires_grad = False
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.weights = layer_weights
        self.loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:

        if self.normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std

        loss = 0.0
        for i, layer in enumerate(self.model):
            input = layer(input)
            target = layer(target)
            if self.weights[i] != 0:
                loss += self.weights[i] * self.loss(input, target)

        return loss


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
