from typing import Sequence

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["Entropy"]


class Entropy(Metric):
    r"""Calculates the mean of `entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_.

    .. math:: H = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C -p_{i,c} \log p_{i,c},
       \quad p_{i,c} = \frac{\exp(z_{i,c})}{\sum_{c'=1}^C \exp(z_{i,c'})}

    where :math:`p_{i,c}` is the prediction probability of :math:`i`-th data belonging to the class :math:`c`.

    - ``update`` must receive output of the form ``(y_pred, y)`` while ``y`` is not used in this metric.
    - ``y_pred`` is expected to be the unnormalized logits for each class. :math:`(B, C)` (classification)
      or :math:`(B, C, ...)` (e.g., image segmentation) shapes are allowed.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = Entropy()
            metric.attach(default_evaluator, 'entropy')
            y_true = torch.tensor([0, 1, 2])  # not considered in the Entropy metric.
            y_pred = torch.tensor([
                [ 0.0000,  0.6931,  1.0986],
                [ 1.3863,  1.6094,  1.6094],
                [ 0.0000, -2.3026, -2.3026]
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['entropy'])

        .. testoutput::

            0.8902875582377116

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    _state_dict_all_req_keys = ("_sum_of_entropies", "_num_examples")

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_entropies = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred = output[0].detach()
        if y_pred.ndim >= 3:
            num_classes = y_pred.shape[1]
            # (B, C, ...) -> (B, ..., C) -> (B*..., C)
            # regarding as B*... predictions
            y_pred = y_pred.movedim(1, -1).reshape(-1, num_classes)
        elif y_pred.ndim == 1:
            raise ValueError(f"y_pred must be in the shape of (B, C) or (B, C, ...), got {y_pred.shape}.")

        prob = F.softmax(y_pred, dim=1)
        log_prob = F.log_softmax(y_pred, dim=1)

        self._update(prob, log_prob)

    def _update(self, prob: torch.Tensor, log_prob: torch.Tensor) -> None:
        entropy_sum = -torch.sum(prob * log_prob)
        self._sum_of_entropies += entropy_sum.to(self._device)
        self._num_examples += prob.shape[0]

    @sync_all_reduce("_sum_of_entropies", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Entropy must have at least one example before it can be computed.")
        return self._sum_of_entropies.item() / self._num_examples
