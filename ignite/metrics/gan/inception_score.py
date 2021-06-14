from typing import Callable, Union

import torch
from torch import exp, log

from ignite.exceptions import NotComputableError

# These decorators helps with distributed settings
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class InceptionScore(Metric):
    r"""Calculates Inception Score.

    .. math::
       \text{IS} = \mod{\text{p(y|x)} * \mod{\text{log(p(y|x))} - \text{log(p(y|x))}}}

    where :math:`p(y|x)` is the conditional probability of image being the given object and
    :math:`p(y)` is the marginal probability that the given image is real.

    More details can be found in `Barratt et al. 2018`__.

    __ https://arxiv.org/pdf/1801.01973.pdf


    Args:
        num_probabilities: number of probabilities prediccted by the model or number of classes of the model
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Example:

        .. code-block:: python

            from ignite.metric.gan.IS import InceptionScore
            import torch

            probabilities = torch.rand(10,2048), torch.rand(10,2048)

            m = InceptionScore(num_probabilities=2048)
            m.update(probabilities)
            print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self, num_probabilities: int, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = "cpu"
    ) -> None:
        if num_probabilities <= 0:
            raise ValueError(f"num of probabilities must be greater to zero, got: {num_probabilities}")
        self._num_probs = num_probabilities
        self._eps = 1e-16
        super(InceptionScore, self).__init__(output_transform=output_transform, device=device)

    @staticmethod
    def _check_feature_input(samples: torch.Tensor) -> None:
        if samples.dim() != 2:
            raise ValueError(f"Probabilities must be a tensor of dim 2, got: {samples.dim()}")
        if samples.shape[0] == 0:
            raise ValueError(f"Batch size should be greater than one, got: {samples.shape[0]}")
        if samples.shape[1] == 0:
            raise ValueError(f"Number of Probabilities should be greater than one, got: {samples.shape[1]}")

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0
        self._prob_total = torch.zeros(self._num_probs, dtype=torch.float64).to(self._device)
        self._total_kl_d = torch.zeros(self._num_probs, dtype=torch.float64).to(self._device)
        super(InceptionScore, self).reset()

    @reinit__is_reduced
    def update(self, samples: torch.Tensor) -> None:
        self._check_feature_input(samples)
        for sample in samples:
            self._num_examples += 1
            self._prob_total += sample.to(self._device)
            self._total_kl_d += sample.to(self._device) * log(sample + self._eps).to(self._device)

    @sync_all_reduce("_num_examples", "_prob_total", "_total_kl_d")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("IS must have at least one example before it can be computed.")
        mean_probs = self._prob_total / self._num_examples
        excess_entropy = self._prob_total * log(mean_probs + self._eps)
        avg_kl_d = sum(self._total_kl_d - excess_entropy) / self._num_examples
        return exp(torch.tensor(avg_kl_d))
