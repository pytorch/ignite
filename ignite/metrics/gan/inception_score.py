from typing import Callable, Optional, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.gan.utils import InceptionModel, _BaseInceptionMetric

# These decorators helps with distributed settings
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

__all__ = ["InceptionScore"]


class InceptionScore(_BaseInceptionMetric):
    r"""Calculates Inception Score.

    .. math::
       \text{IS(G)} = \exp(\frac{1}{N}\sum_{i=1}^{N} D_{KL} (p(y|x^{(i)} \parallel \hat{p}(y))))

    where :math:`p(y|x)` is the conditional probability of image being the given object and
    :math:`p(y)` is the marginal probability that the given image is real, `G` refers to the
    generated image and :math:`D_{KL}` refers to KL Divergence of the above mentioned probabilities.

    More details can be found in `Barratt et al. 2018`__.

    __ https://arxiv.org/pdf/1801.01973.pdf

    .. note::
        The default Inception model requires the `torchvision` module to be installed.

    Args:
        num_features: number of features predicted by the model or number of classes of the model. Default
            value is 1000.
        feature_extractor: a torch Module for predicting the probabilities from the input data.
            It returns a tensor of shape (batch_size, num_features).
            If neither ``num_features`` nor ``feature_extractor`` are defined, by default we use an ImageNet
            pretrained Inception Model. If only ``num_features`` is defined but ``feature_extractor`` is not
            defined, ``feature_extractor`` is assigned Identity Function.
            Please note that the class object will be implicitly converted to device mentioned in the
            ``device`` argument.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``y_pred``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Example:

        .. code-block:: python

            from ignite.metric.gan import InceptionScore
            import torch

            images = torch.rand(10, 3, 299, 299)

            m = InceptionScore()
            m.update(images)
            print(m.compute())

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._eps = 1e-16

        super(InceptionScore, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    @reinit__is_reduced
    def reset(self) -> None:

        self._num_examples = 0

        self._prob_total = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)
        self._total_kl_d = torch.zeros(self._num_features, dtype=torch.float64, device=self._device)

        super(InceptionScore, self).reset()

    @reinit__is_reduced
    def update(self, output: torch.Tensor) -> None:

        probabilities = self._extract_features(output)

        prob_sum = torch.sum(probabilities, 0, dtype=torch.float64)
        log_prob = torch.log(probabilities + self._eps)
        if log_prob.dtype != probabilities.dtype:
            log_prob = log_prob.to(probabilities)
        kl_sum = torch.sum(probabilities * log_prob, 0, dtype=torch.float64)

        self._num_examples += probabilities.shape[0]
        self._prob_total += prob_sum
        self._total_kl_d += kl_sum

    @sync_all_reduce("_num_examples", "_prob_total", "_total_kl_d")
    def compute(self) -> float:

        if self._num_examples == 0:
            raise NotComputableError("InceptionScore must have at least one example before it can be computed.")

        mean_probs = self._prob_total / self._num_examples
        log_mean_probs = torch.log(mean_probs + self._eps)
        if log_mean_probs.dtype != self._prob_total.dtype:
            log_mean_probs = log_mean_probs.to(self._prob_total)
        excess_entropy = self._prob_total * log_mean_probs
        avg_kl_d = torch.sum(self._total_kl_d - excess_entropy) / self._num_examples

        return torch.exp(avg_kl_d).item()
