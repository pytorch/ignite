from typing import Callable, List, Union

import torch
import torchvision

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize: bool = True) -> None:
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, feature_layers: List = [0, 1, 2, 3], style_layers: List = []
    ) -> torch.Tensor:
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = self.transform(target, mode="bilinear", size=(224, 224), align_corners=False)
        loss = torch.tensor(0.0)
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
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
