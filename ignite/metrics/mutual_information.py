import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Entropy
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

__all__ = ["MutualInformation"]


class MutualInformation(Entropy):
    r"""Calculates the `mutual information <https://en.wikipedia.org/wiki/Mutual_information>`_
    between input :math:`X` and prediction :math:`Y`.

    .. math::
       \begin{align*}
            I(X;Y) &= H(Y) - H(Y|X) = H \left( \frac{1}{N}\sum_{i=1}^N \hat{\mathbf{p}}_i \right)
            - \frac{1}{N}\sum_{i=1}^N H(\hat{\mathbf{p}}_i), \\
            H(\mathbf{p}) &= -\sum_{c=1}^C p_c \log p_c.
       \end{align*}

    where :math:`\hat{\mathbf{p}}_i` is the prediction probability vector for :math:`i`-th input,
    and :math:`H(\mathbf{p})` is the entropy of :math:`\mathbf{p}`.

    Intuitively, this metric measures how well input data are clustered by classes in the feature space [1].

    [1] https://proceedings.mlr.press/v70/hu17b.html

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

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = MutualInformation()
            metric.attach(default_evaluator, 'mutual_information')
            y_true = torch.tensor([0, 1, 2])  # not considered in the MutualInformation metric.
            y_pred = torch.tensor([
                [ 0.0000,  0.6931,  1.0986],
                [ 1.3863,  1.6094,  1.6094],
                [ 0.0000, -2.3026, -2.3026]
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['mutual_information'])

        .. testoutput::

           0.18599730730056763
    """

    _state_dict_all_req_keys = ("_sum_of_probabilities",)

    @reinit__is_reduced
    def reset(self) -> None:
        super().reset()
        self._sum_of_probabilities = torch.tensor(0.0, device=self._device)

    def _update(self, prob: torch.Tensor, log_prob: torch.Tensor) -> None:
        super()._update(prob, log_prob)
        # We can't use += below as _sum_of_probabilities can be a scalar and prob.sum(dim=0) is a vector
        self._sum_of_probabilities = self._sum_of_probabilities + prob.sum(dim=0).to(self._device)

    @sync_all_reduce("_sum_of_probabilities", "_sum_of_entropies", "_num_examples")
    def compute(self) -> float:
        n = self._num_examples
        if n == 0:
            raise NotComputableError("MutualInformation must have at least one example before it can be computed.")

        marginal_prob = self._sum_of_probabilities / n
        marginal_ent = -(marginal_prob * torch.log(marginal_prob)).sum()
        conditional_ent = self._sum_of_entropies / n
        mi = marginal_ent - conditional_ent
        mi = torch.clamp(mi, min=0.0)  # mutual information cannot be negative
        return float(mi.item())
