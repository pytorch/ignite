from typing import List, Tuple, cast

import torch

import ignite.distributed as idist
from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced


class GeometricMeanRelativeAbsoluteError(_BaseRegression):
    r"""Calculates the Geometric Mean Relative Absolute Error.

    .. math::
        \text{GMRAE} = \exp(\frac{1}{n}\sum_{j=1}^n \ln\frac{|A_j - P_j|}{|A_j - \bar{A}|})

    where :math:`A_j` is the ground truth, :math:`P_j` is the predicted value
    and :math: `bar{A}` is the mean of the ground truth.

    More details can be found in `Botchkarev 2018`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006

    Parameters are inherited from ``Metric.__init__``.

    .. warning::

        Current implementation of GMRAE stores all input data (output and target)
        as tensors before computing the metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.

        In distributed configuration, all stored data (output and target) is mutually collected across all processes
        using all gather collective operation. This can potentially lead to a memory error.

        Compute method compute the metric on zero rank process only and final result is broadcasted to
        all processes.

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
        The output of the engine's ``process_function`` needs to be in format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

        .. testcode::

            metric = GeometricMeanRelativeAbsoluteError()
            metric.attach(default_evaluator, 'gmare')
            y_true = torch.Tensor([0, 1, 2, 3, 4, 5])
            y_pred = y_true * 0.75
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['gmare'])

        .. testoutput::

            0.0...
    """

    @reinit__is_reduced
    def reset(self) -> None:
        self._predictions = []  # type: List[torch.Tensor]
        self._targets = []  # type: List[torch.Tensor]

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        y_pred = y_pred.clone().to(self._device)
        y = y.clone().to(self._device)

        self._predictions.append(y_pred)
        self._targets.append(y)

    def compute(self) -> float:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError(
                "GeometricMeanRelativeAbsoluteError must have at least one example before it can be computed."
            )

        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        # All gather across all processes
        _prediction_tensor = cast(torch.Tensor, idist.all_gather(_prediction_tensor))
        _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))

        result = torch.exp(
            torch.log(
                torch.abs(_target_tensor - _prediction_tensor) / torch.abs(_target_tensor - _target_tensor.mean())
            ).mean()
        ).item()

        return result
