from typing import Callable, List, Optional, Union, cast

import torch
from torch import nn
from torch.nn import functional as F

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced

__all__ = ["InceptionScore"]


class InceptionScore(Metric):
    r"""Calculates the Inception Score of a GAN model

    More details can be found in `Improved Techniques for Training GANs`__.

    __ https://arxiv.org/pdf/1606.03498.pdf


    - ``update`` must receive output of the form ``y_pred`` or ``{'y_pred': y_pred}``.
    - ``y_pred`` should have the following shape (batch_size, ) and contains syntheic images created by the generator.


  .. warning::

        Current implementation stores all input data before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.

        In distributed configuration, all stored data is mutually collected across all processes
        using all gather collective operation. This can potentially lead to a memory error.

    Args:
        splits: number of splits to calculate the mean inception score.
        inception_model: model used for extracting class probabilities from images
        If not specified, InceptionV3 Pretrained on ImageNet will be used.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
   Example:

    .. code-block:: python

        from ignite.metrics.gan import InceptionScore

        m = InceptionScore(splits=4)

        y_pred = torch.rand((16,3,299,299))  # size must match the input shape of the InceptionV3 model

        m.update(y_pred)

        print(m.compute())

    """

    def __init__(
        self,
        splits: int = 10,
        inception_model: Optional[nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if inception_model is None:
            try:
                from torchvision import models  # type: ignore

                inception_model = models.inception_v3(pretrained=True, transform_input=False)
            except ImportError:
                raise ValueError("Argument inception_model should be set")
        super(InceptionScore, self).__init__(output_transform=output_transform, device=device)
        self.inception_model = inception_model.eval().to(self._device)
        self.n_splits = splits

    @reinit__is_reduced
    def reset(self) -> None:
        self._probs = []  # type: List[torch.Tensor]

    @reinit__is_reduced
    def update(self, output: torch.Tensor) -> None:
        generated = output.detach()
        inception_output = self.inception_model(generated)
        probs = F.softmax(inception_output)
        probs = probs.clone().to(self._device)
        self._probs.append(probs)

    def compute(self) -> float:
        if len(self._probs) < 1:
            raise NotComputableError("Inception score must have at least one example before it can be computed.")

        ws = idist.get_world_size()
        _probs_tensor = torch.cat(self._probs, dim=0)

        if ws > 1 and not self._is_reduced:
            _probs_tensor = cast(torch.Tensor, idist.all_gather(_probs_tensor))
        self._is_reduced = True

        result = 0.0
        if idist.get_rank() == 0:
            N = _probs_tensor.shape[0]
            scores = torch.zeros((self.n_splits,))
            for i in range(self.n_splits):
                part = _probs_tensor[i * (N // self.n_splits) : (i + 1) * (N // self.n_splits)]
                kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0)))
                kl = torch.mean(torch.sum(kl, dim=1))
                scores[i] = torch.exp(kl)
            result = torch.mean(scores).item()
        if ws > 1:
            result = cast(float, idist.broadcast(result, src=0))

        return result
