import warnings
from typing import Callable, cast, List, Optional, Sequence, Tuple, Union

import torch
from typing_extensions import Literal

import ignite.distributed as idist
from ignite.distributed.utils import all_gather_tensors_with_shapes
from ignite.metrics.metric import Metric, reinit__is_reduced
from ignite.metrics.precision import _BaseClassification
from ignite.utils import to_onehot


class _BaseAveragePrecision:
    def __init__(
        self,
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        class_mean: Optional[Literal["micro", "macro", "weighted"]] = "macro",
    ) -> None:
        r"""Base class for Average Precision metric.

        This class contains the methods for setting up the thresholds and computing AP & AR.

        Args:
            rec_thresholds: recall thresholds (sensivity levels) to be considered for computing Mean Average Precision.
                It could be a 1-dim tensor or a sequence of floats. Its values should be between 0 and 1 and don't need
                to be sorted. If missing, thresholds are considered automatically using the data.
            class_mean: how to compute mean of the average precision across classes or incorporate class
                dimension into computing precision. It's ignored in binary classification. Available options are

                None
                  An 1-dimensional tensor of mean (taken across additional mean dimensions) average precision per class
                  is returned. If there's no ground truth sample for a class, ``0`` is returned for that.

                'micro'
                  Precision is computed counting stats of classes/labels altogether. This option
                  incorporates class in the very precision measurement.

                  .. math::
                      \text{Micro P} = \frac{\sum_{c=1}^C TP_c}{\sum_{c=1}^C TP_c+FP_c}

                  where :math:`C` is the number of classes/labels. :math:`c` in :math:`TP_c`
                  and :math:`FP_c` means that the terms are computed for class/label :math:`c` (in a one-vs-rest
                  sense in multiclass case).

                  For multiclass inputs, this is equivalent with mean average accuracy.

                'weighted'
                  like macro but considers class/label imbalance. For multiclass input,
                  it computes AP for each class then returns mean of them weighted by
                  support of classes (number of actual samples in each class). For multilabel input,
                  it computes AP for each label then returns mean of them weighted by support
                  of labels (number of actual positive samples in each label).

                'macro'
                  computes macro precision which is unweighted mean of AP computed across classes/labels. Default.
        """
        if rec_thresholds is not None:
            self.rec_thresholds: Optional[torch.Tensor] = self._setup_thresholds(rec_thresholds, "rec_thresholds")
        else:
            self.rec_thresholds = None

        if class_mean is not None and class_mean not in ("micro", "macro", "weighted"):
            raise ValueError(f"Wrong `class_mean` parameter, given {class_mean}")
        self.class_mean = class_mean

    def _setup_thresholds(self, thresholds: Union[Sequence[float], torch.Tensor], threshold_type: str) -> torch.Tensor:
        if isinstance(thresholds, Sequence):
            thresholds = torch.tensor(thresholds, dtype=torch.double)

        if isinstance(thresholds, torch.Tensor):
            if thresholds.ndim != 1:
                raise ValueError(
                    f"{threshold_type} should be a one-dimensional tensor or a sequence of floats"
                    f", given a {thresholds.ndim}-dimensional tensor."
                )
            thresholds = thresholds.sort().values
        else:
            raise TypeError(f"{threshold_type} should be a sequence of floats or a tensor, given {type(thresholds)}.")

        if min(thresholds) < 0 or max(thresholds) > 1:
            raise ValueError(f"{threshold_type} values should be between 0 and 1, given {thresholds}")

        return thresholds

    def _compute_average_precision(self, recall: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        """Measuring average precision.

        Args:
            recall: n-dimensional tensor whose last dimension represents confidence thresholds as much as #samples.
            Should be ordered in ascending order in its last dimension.
            precision: like ``recall`` in the shape.

        Returns:
            average_precision: (n-1)-dimensional tensor containing the average precisions.
        """
        if self.rec_thresholds is not None:
            rec_thresholds = self.rec_thresholds.repeat((*recall.shape[:-1], 1))
            rec_thresh_indices = torch.searchsorted(recall, rec_thresholds)
            precision = precision.take_along_dim(
                rec_thresh_indices.where(rec_thresh_indices != recall.size(-1), 0), dim=-1
            ).where(rec_thresh_indices != recall.size(-1), 0)
            recall = rec_thresholds
        recall_differential = recall.diff(
            dim=-1, prepend=torch.zeros((*recall.shape[:-1], 1), device=recall.device, dtype=recall.dtype)
        )
        return torch.sum(recall_differential * precision, dim=-1)


def _cat_and_agg_tensors(
    tensors: List[torch.Tensor],
    tensor_shape_except_last_dim: Tuple[int],
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """
    Concatenate tensors in ``tensors`` at their last dimension and gather all tensors from across all processes.
    All tensors should have the same shape (denoted by ``tensor_shape_except_last_dim``) except at their
    last dimension.
    """
    num_preds = torch.tensor(
        [sum([tensor.shape[-1] for tensor in tensors]) if tensors else 0],
        device=device,
    )
    all_num_preds = cast(torch.Tensor, idist.all_gather(num_preds)).tolist()
    tensor = (
        torch.cat(tensors, dim=-1)
        if tensors
        else torch.empty((*tensor_shape_except_last_dim, 0), dtype=dtype, device=device)
    )
    shape_across_ranks = [(*tensor_shape_except_last_dim, num_pred_in_rank) for num_pred_in_rank in all_num_preds]
    return torch.cat(
        all_gather_tensors_with_shapes(
            tensor,
            shape_across_ranks,
        ),
        dim=-1,
    )


class MeanAveragePrecision(_BaseClassification, _BaseAveragePrecision):
    _y_pred: List[torch.Tensor]
    _y_true: List[torch.Tensor]

    def __init__(
        self,
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        class_mean: Optional["Literal['micro', 'macro', 'weighted']"] = "macro",
        is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        r"""Calculate the mean average precision metric i.e. mean of the averaged-over-recall precision for
        classification task:

        .. math::
            \text{Average Precision} = \sum_{k=1}^{\#rec\_thresholds} (r_k - r_{k-1}) P_k

        Mean average precision attempts to give a measure of detector or classifier precision at various
        sensivity levels a.k.a recall thresholds. This is done by summing precisions at different recall
        thresholds weighted by the change in recall, as if the area under precision-recall curve is being computed.
        Mean average precision is then computed by taking the mean of this average precision over different classes.

        All the binary, multiclass and multilabel data are supported. In the latter case,
        ``is_multilabel`` should be set to true.

        `mean` in the mean average precision accounts for mean of the average precision across classes. ``class_mean``
        determines how to take this mean.

        Args:
            rec_thresholds: recall thresholds (sensivity levels) to be considered for computing Mean Average Precision.
                It could be a 1-dim tensor or a sequence of floats. Its values should be between 0 and 1 and don't need
                to be sorted. If missing, thresholds are considered automatically using the data.
            class_mean: how to compute mean of the average precision across classes or incorporate class
                dimension into computing precision. It's ignored in binary classification. Available options are

                None
                  A 1-dimensional tensor of mean (taken across additional mean dimensions) average precision per class
                  is returned. If there's no ground truth sample for a class, ``0`` is returned for that.

                'micro'
                  Precision is computed counting stats of classes/labels altogether. This option
                  incorporates class in the very precision measurement.

                  .. math::
                      \text{Micro P} = \frac{\sum_{c=1}^C TP_c}{\sum_{c=1}^C TP_c+FP_c}

                  where :math:`C` is the number of classes/labels. :math:`c` in :math:`TP_c`
                  and :math:`FP_c` means that the terms are computed for class/label :math:`c` (in a one-vs-rest
                  sense in multiclass case).

                  For multiclass inputs, this is equivalent to mean average accuracy.

                'weighted'
                  like macro but considers class/label imbalance. For multiclass input,
                  it computes AP for each class then returns mean of them weighted by
                  support of classes (number of actual samples in each class). For multilabel input,
                  it computes AP for each label then returns mean of them weighted by support
                  of labels (number of actual positive samples in each label).

                'macro'
                  computes macro precision which is unweighted mean of AP computed across classes/labels. Default.

            is_multilabel: determines if the data is multilabel or not. Default False.
            output_transform: a callable that is used to transform the
                :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs. This metric requires the output
                as ``(y_pred, y)``.
            device: specifies which device updates are accumulated on. Setting the
                metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
                non-blocking. By default, CPU.
            skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
                true for multi-output model, for example, if ``y_pred`` and ``y`` contain multi-ouput as
                ``(y_pred_a, y_pred_b)`` and ``(y_a, y_b)``, in which case the update method is called for
                ``(y_pred_a, y_a)`` and ``(y_pred_b, y_b)``.Alternatively, ``output_transform`` can be used to handle
                this.

        .. versionadded:: 0.5.2
        """

        super(MeanAveragePrecision, self).__init__(
            output_transform=output_transform,
            is_multilabel=is_multilabel,
            device=device,
            skip_unrolling=skip_unrolling,
        )
        super(Metric, self).__init__(rec_thresholds=rec_thresholds, class_mean=class_mean)

    @reinit__is_reduced
    def reset(self) -> None:
        """
        Reset method of the metric
        """
        super().reset()
        self._y_pred = []
        self._y_true = []

    def _check_binary_multilabel_cases(self, output: Sequence[torch.Tensor]) -> None:
        # Ignore the check in `_BaseClassification` since `y_pred` consists of probabilities here.
        _, y = output
        if not torch.equal(y, y**2):
            raise ValueError("For binary cases, y must be comprised of 0's and 1's.")

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        super()._check_type(output)
        y_pred, y = output
        if y_pred.dtype in (torch.int, torch.long):
            raise TypeError(f"`y_pred` should be a float tensor, given {y_pred.dtype}")
        if self._type == "multiclass" and y.dtype != torch.long:
            warnings.warn("`y` should be of dtype long when entry type is multiclass", RuntimeWarning)

    def _prepare_output(self, output: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Prepares and returns ``y_pred`` and ``y`` tensors. Input and output shapes of the method is as follows.
        ``C`` and ``L`` denote the number of classes and labels in multiclass and multilabel inputs respectively.

        ========== =========== ============
        ``y_pred``
        -----------------------------------
        Data type  Input shape Output shape
        ========== =========== ============
        Binary     (N, ...)    (1, N * ...)
        Multilabel (N, L, ...) (L, N * ...)
        Multiclass (N, C, ...) (C, N * ...)
        ========== =========== ============

        ========== =========== ============
        ``y``
        -----------------------------------
        Data type  Input shape Output shape
        ========== =========== ============
        Binary     (N, ...)    (1, N * ...)
        Multilabel (N, L, ...) (L, N * ...)
        Multiclass (N, ...)    (N * ...)
        ========== =========== ============
        """
        y_pred, y = output[0].detach(), output[1].detach()

        if self._type == "multilabel":
            num_classes = y_pred.size(1)
            yp = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
            yt = torch.transpose(y, 1, 0).reshape(num_classes, -1)
        elif self._type == "binary":
            yp = y_pred.view(1, -1)
            yt = y.view(1, -1)
        else:  # Multiclass
            num_classes = y_pred.size(1)
            if y.max() + 1 > num_classes:
                raise ValueError(
                    f"y_pred contains fewer classes than y. Number of classes in prediction is {num_classes}"
                    f" and an element in y has invalid class = {y.max().item() + 1}."
                )
            yt = y.view(-1)
            yp = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)

        return yp, yt

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Metric update function using prediction and target.

        Args:
            output: a binary tuple consisting of prediction and target tensors

                This metric follows the same rules on ``output`` members shape as the
                :meth:`Precision.update <.metrics.precision.Precision.update>` except for ``y_pred`` of binary and
                multilabel data which should be comprised of positive class probabilities here.
        """
        self._check_shape(output)
        self._check_type(output)
        yp, yt = self._prepare_output(output)
        self._y_pred.append(yp.to(self._device))
        self._y_true.append(yt.to(self._device, dtype=torch.uint8 if self._type != "multiclass" else torch.long))

    def _compute_recall_and_precision(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, y_true_positive_count: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Measuring recall & precision.

        Shape of function inputs and return values follow the table below.
        N is the number of samples. \#unique scores represents number of
        unique scores in ``scores`` which is actually the number of thresholds.

        ===================== =======================================
        **Object**            **Shape**
        ===================== =======================================
        y_true                (N,)
        y_pred                (N,)
        y_true_positive_count () (A single float)
        recall                (\#unique scores,)
        precision             (\#unique scores,)
        ===================== =======================================

        Returns:
            `(recall, precision)`
        """
        indices = torch.argsort(y_pred, stable=True, descending=True)
        tp_summation = y_true[indices].cumsum(dim=0)
        if tp_summation.device != torch.device("mps"):
            tp_summation = tp_summation.double()

        # Adopted from Scikit-learn's implementation
        unique_scores_indices = torch.nonzero(
            y_pred[indices].diff(append=(y_pred.max() + 1).unsqueeze(dim=0)), as_tuple=True
        )[0]
        tp_summation = tp_summation[..., unique_scores_indices]
        fp_summation = (unique_scores_indices + 1) - tp_summation

        if y_true_positive_count == 0:
            # To be aligned with Scikit-Learn
            recall = torch.ones_like(tp_summation, device=self._device, dtype=torch.float)
        else:
            recall = tp_summation / y_true_positive_count

        predicted_positive = tp_summation + fp_summation
        precision = tp_summation / torch.where(predicted_positive == 0, 1, predicted_positive)
        return recall, precision

    def compute(self) -> Union[torch.Tensor, float]:
        """
        Compute method of the metric
        """
        if self._num_classes is None:
            raise RuntimeError("Metric could not be computed without any update method call")
        num_classes = self._num_classes

        y_true = _cat_and_agg_tensors(
            self._y_true,
            cast(Tuple[int], ()) if self._type == "multiclass" else (num_classes,),
            torch.long if self._type == "multiclass" else torch.uint8,
            self._device,
        )
        fp_precision = torch.double if self._device != torch.device("mps") else torch.float32
        y_pred = _cat_and_agg_tensors(self._y_pred, (num_classes,), fp_precision, self._device)

        if self._type == "multiclass":
            y_true = to_onehot(y_true, num_classes=num_classes).T
        if self.class_mean == "micro":
            y_true = y_true.reshape(1, -1)
            y_pred = y_pred.view(1, -1)
        y_true_positive_count = y_true.sum(dim=-1)
        average_precisions = torch.zeros_like(y_true_positive_count, device=self._device, dtype=fp_precision)
        for cls in range(y_true_positive_count.size(0)):
            recall, precision = self._compute_recall_and_precision(y_true[cls], y_pred[cls], y_true_positive_count[cls])
            average_precisions[cls] = self._compute_average_precision(recall, precision)
        if self._type == "binary":
            return average_precisions.item()
        if self.class_mean is None:
            return average_precisions
        elif self.class_mean == "weighted":
            return torch.sum(y_true_positive_count * average_precisions) / y_true_positive_count.sum()
        else:
            return average_precisions.mean()
