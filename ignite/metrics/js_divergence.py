import torch
import torch.nn.functional as F
from packaging.version import Version

from ignite.exceptions import NotComputableError
from ignite.metrics.kl_divergence import KLDivergence
from ignite.metrics.metric import sync_all_reduce

__all__ = ["JSDivergence"]

TORCH_VERSION_GE_160 = Version(torch.__version__) >= Version("1.6.0")


class JSDivergence(KLDivergence):
    r"""Calculates the mean of `Jensen-Shannon (JS) divergence
    <https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence>`_.

    .. math::
       \begin{align*}
           D_\text{JS}(\mathbf{p}_i \| \mathbf{q}_i) &= \frac{1}{2} D_\text{KL}(\mathbf{p}_i \| \mathbf{m}_i)
           + \frac{1}{2} D_\text{KL}(\mathbf{q}_i \| \mathbf{m}_i), \\
           \mathbf{m}_i &= \frac{1}{2}(\mathbf{p}_i + \mathbf{q}_i), \\
           D_\text{KL}(\mathbf{p}_i \| \mathbf{q}_i) &= \sum_{c=1}^C p_{i,c} \log \frac{p_{i,c}}{q_{i,c}}.
       \end{align*}

    where :math:`\mathbf{p}_i` and :math:`\mathbf{q}_i` are the ground truth and prediction probability tensors,
    and :math:`D_\text{KL}` is the KL-divergence.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` and ``y`` are expected to be the unnormalized logits for each class. :math:`(B, C)` (classification)
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

            metric = JSDivergence()
            metric.attach(default_evaluator, 'js-div')
            y_true = torch.tensor([
                [ 0.0000, -2.3026, -2.3026],
                [ 1.3863,  1.6094,  1.6094],
                [ 0.0000,  0.6931,  1.0986]
            ])
            y_pred = torch.tensor([
                [ 0.0000,  0.6931,  1.0986],
                [ 1.3863,  1.6094,  1.6094],
                [ 0.0000, -2.3026, -2.3026]
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['js-div'])

        .. testoutput::

           0.16266516844431558

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    def _update(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        y_pred_prob = F.softmax(y_pred, dim=1)
        y_prob = F.softmax(y, dim=1)
        m_prob = (y_pred_prob + y_prob) / 2
        m_log = m_prob.log()

        if TORCH_VERSION_GE_160:
            # log_target option can be used from 1.6.0
            y_pred_log = F.log_softmax(y_pred, dim=1)
            y_log = F.log_softmax(y, dim=1)
            self._sum_of_kl += (
                F.kl_div(m_log, y_pred_log, log_target=True, reduction="sum")
                + F.kl_div(m_log, y_log, log_target=True, reduction="sum")
            ).to(self._device)
        else:
            # y_pred and y are expected to be probabilities
            self._sum_of_kl += (
                F.kl_div(m_log, y_pred_prob, reduction="sum") + F.kl_div(m_log, y_prob, reduction="sum")
            ).to(self._device)

    @sync_all_reduce("_sum_of_kl", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("JSDivergence must have at least one example before it can be computed.")
        return self._sum_of_kl.item() / (self._num_examples * 2)
