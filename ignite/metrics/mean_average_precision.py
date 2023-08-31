import warnings
from typing import Callable, cast, List, Optional, Sequence, Tuple, Union

import torch
from typing_extensions import Literal

import ignite.distributed as idist
from ignite.distributed.utils import all_gather_tensors_with_shapes
from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.recall import _BasePrecisionRecall
from ignite.utils import to_onehot


class _BaseMeanAveragePrecision(_BasePrecisionRecall):
    def __init__(
        self,
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        average: Optional[str] = "precision",
        class_mean: Optional[Literal["micro", "macro", "weighted", "with_other_dims"]] = "macro",
        is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        r"""Base class for Mean Average Precision in classification and detection tasks.

        Mean average precision is computed by taking the mean of the average precision over different classes
        and possibly some additional dimensions in the detection task. ``class_mean`` determines how to take this mean.
        In the detection tasks, it's possible to take the mean in other respects as well e.g. IoU threshold in an
        object detection task.

        Args:
            rec_thresholds: recall thresholds (sensivity levels) to be considered for computing Mean Average Precision.
                It could be a 1-dim tensor or a sequence of floats. Its values should be between 0 and 1 and don't need
                to be sorted. If missing, thresholds are considered automatically using the data.
            average: one of values precision or max-precision. In the former case, the precision at a
                recall threshold is used for that threshold:

                .. math::
                    \text{Average Precision} = \sum_{k=1}^{\#rec\_thresholds} (r_k - r_{k-1}) P_k

                :math:`r` stands for recall thresholds and :math:`P` for precision values. :math:`r_0` is set to zero.

                In the latter case, the maximum precision across thresholds greater or equal a recall threshold is
                considered as the summation operand; In other words, the precision peek across lower or equal
                sensivity levels is used for a recall threshold:

                .. math::
                    \text{Average Precision} = \sum_{k=1}^{\#rec\_thresholds} (r_k - r_{k-1}) max(P_{k:})

                Default is "precision".
            class_mean: how to compute mean of the average precision across classes or incorporate class
                dimension into computing precision. It's ignored in binary classification. Available options are

                None
                  An 1-dimensional tensor of mean (taken across additional mean dimensions) average precision per class
                  is returned. If there's no ground truth sample for a class, ``0`` is returned for that.

                micro
                  Precision is computed counting stats of classes/labels altogether. This option
                  incorporates class in the very precision measurement.

                  .. math::
                      \text{Micro P} = \frac{\sum_{c=1}^C TP_c}{\sum_{c=1}^C TP_c+FP_c}

                  where :math:`C` is the number of classes/labels. :math:`c` in :math:`TP_c`
                  and :math:`FP_c` means that the terms are computed for class/label :math:`c` (in a one-vs-rest
                  sense in multiclass case).

                  For multiclass inputs, this is equivalent with mean average accuracy.

                weighted
                  like macro but considers class/label imbalance. For multiclass input,
                  it computes AP for each class then returns mean of them weighted by
                  support of classes (number of actual samples in each class). For multilabel input,
                  it computes AP for each label then returns mean of them weighted by support
                  of labels (number of actual positive samples in each label).

                'macro'
                  computes macro precision which is unweighted mean of AP computed across classes/labels. Default.

                'with_other_dims'
                  Mean over class dimension is taken with additional mean dimensions all at once, despite macro and
                  weighted in which mean over additional dimensions is taken beforehand. Only available in detection.

                Note:
                    Please note that classes with no ground truth are not considered into the mean in detection.

            is_multilabel: Used in classification task and determines if the data
                is multilabel or not. Default False.
            output_transform: a callable that is used to transform the
                :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs. This metric requires the output
                as ``(y_pred, y)``.
            device: specifies which device updates are accumulated on. Setting the
                metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
                non-blocking. By default, CPU.
        """
        if rec_thresholds is not None:
            self.rec_thresholds: Optional[torch.Tensor] = self._setup_thresholds(rec_thresholds, "rec_thresholds")
        else:
            self.rec_thresholds = None

        if average not in ("precision", "max-precision"):
            raise ValueError(f"Wrong `average` parameter, given {average}")
        self.average = average

        if class_mean is not None and class_mean not in ("micro", "macro", "weighted", "with_other_dims"):
            raise ValueError(f"Wrong `class_mean` parameter, given {class_mean}")
        self.class_mean = class_mean

        super().__init__(output_transform=output_transform, is_multilabel=is_multilabel, device=device)

    def _setup_thresholds(self, thresholds: Union[Sequence[float], torch.Tensor], threshold_type: str) -> torch.Tensor:
        if isinstance(thresholds, Sequence):
            thresholds = torch.tensor(thresholds)

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

        return cast(torch.Tensor, thresholds)

    def _compute_average_precision(self, recall: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        """Measuring average precision which is the common operation among different settings of the metric.

        Args:
            recall: n-dimensional tensor whose last dimension is the dimension of the samples. Should be ordered in
                ascending order in its last dimension.
            precision: like ``recall`` in the shape.

        Returns:
            average_precision: (n-1)-dimensional tensor containing the average precision for mean dimensions.
        """
        precision_integrand = (
            precision.flip(-1).cummax(dim=-1).values.flip(-1) if self.average == "max-precision" else precision
        )
        if self.rec_thresholds is not None:
            rec_thresholds = self.rec_thresholds.repeat((*recall.shape[:-1], 1))
            rec_thresh_indices = torch.searchsorted(recall, rec_thresholds)
            precision_integrand = precision_integrand.take_along_dim(
                rec_thresh_indices.where(rec_thresh_indices != recall.size(-1), 0), dim=-1
            ).where(rec_thresh_indices != recall.size(-1), 0)
            recall = rec_thresholds
        recall_differential = recall.diff(
            dim=-1, prepend=torch.zeros((*recall.shape[:-1], 1), device=self._device, dtype=torch.double)
        )
        return torch.sum(recall_differential * precision_integrand, dim=-1)


class MeanAveragePrecision(_BaseMeanAveragePrecision):
    _scores: List[torch.Tensor]
    _P: List[torch.Tensor]

    def __init__(
        self,
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        average: Optional["Literal['precision', 'max-precision']"] = "precision",
        class_mean: Optional["Literal['micro', 'macro', 'weighted']"] = "macro",
        is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        r"""Calculate the mean average precision metric i.e. mean of the averaged-over-recall precision for
        classification task.

        Mean average precision attempts to give a measure of detector or classifier precision at various
        sensivity levels a.k.a recall thresholds. This is done by summing precisions at different recall
        thresholds weighted by the change in recall, as if the area under precision-recall curve is being computed.
        Mean average precision is then computed by taking the mean of this average precision over different classes.

        For detection tasks, user should use downstream metrics like
        :class:`~ignite.metrics.vision.object_detection_map.ObjectDetectionMAP`. For classification, all the binary,
        multiclass and multilabel data are supported. In the latter case, ``classification_is_multilabel`` should be
        set to true.

        `mean` in the mean average precision accounts for mean of the average precision across classes. ``class_mean``
        determines how to take this mean.

        Args:
            rec_thresholds: recall thresholds (sensivity levels) to be considered for computing Mean Average Precision.
                It could be a 1-dim tensor or a sequence of floats. Its values should be between 0 and 1 and don't need
                to be sorted. If missing, thresholds are considered automatically using the data.
            average: one of values "precision" or "max-precision". In the former case, the precision at a
                recall threshold is used for that threshold:

                .. math::
                    \text{Average Precision} = \sum_{k=1}^{\#rec\_thresholds} (r_k - r_{k-1}) P_k

                :math:`r` stands for recall thresholds and :math:`P` for precision values. :math:`r_0` is set to zero.

                In the latter case, the maximum precision across thresholds greater or equal a recall threshold is
                considered as the summation operand; In other words, the precision peek across lower or equal
                sensivity levels is used for a recall threshold:

                .. math::
                    \text{Average Precision} = \sum_{k=1}^{\#rec\_thresholds} (r_k - r_{k-1}) max(P_{k:})

                Default is "precision".
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
        """

        super().__init__(
            rec_thresholds=rec_thresholds,
            average=average,
            class_mean=class_mean,
            output_transform=output_transform,
            is_multilabel=is_multilabel,
            device=device,
        )

        if self.class_mean == "with_other_dims":
            raise ValueError("class_mean 'with_other_dims' is not compatible with this class.")

    @reinit__is_reduced
    def reset(self) -> None:
        """
        Reset method of the metric
        """
        super(_BasePrecisionRecall, self).reset()
        self._scores = []
        self._P = []

    def _check_binary_multilabel_cases(self, output: Sequence[torch.Tensor]) -> None:
        # Ignore the check in `_BaseClassification` since `y_pred` consists of probabilities here.
        _, y = output
        if not torch.equal(y, y**2):
            raise ValueError("For binary cases, y must be comprised of 0's and 1's.")

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        super(_BasePrecisionRecall, self)._check_type(output)
        y_pred, y = output
        if y_pred.dtype in (torch.int, torch.long):
            raise TypeError(f"`y_pred` should be a float tensor, given {y_pred.dtype}")
        if self._type == "multiclass" and y.dtype != torch.long:
            warnings.warn("`y` should be of dtype long when entry type is multiclass", RuntimeWarning)

    def _prepare_output(self, output: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Prepares and returns scores and P tensor. Input and output shapes of the method is as follows.

        ========== =========== ============
        ``y_pred``
        -----------------------------------
        Data type  Input shape Output shape
        ========== =========== ============
        Binary     (N, ...)    (1, N * ...)
        Multilabel (N, C, ...) (C, N * ...)
        Multiclass (N, C, ...) (C, N * ...)
        ========== =========== ============

        ========== =========== ============
        ``y``
        -----------------------------------
        Data type  Input shape Output shape
        ========== =========== ============
        Binary     (N, ...)    (1, N * ...)
        Multilabel (N, C, ...) (C, N * ...)
        Multiclass (N, ...)    (N * ...)
        ========== =========== ============
        """
        y_pred, y = output[0].detach(), output[1].detach()

        if self._type == "multilabel":
            num_classes = y_pred.size(1)
            scores = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
            P = torch.transpose(y, 1, 0).reshape(num_classes, -1)
        elif self._type == "binary":
            P = y.view(1, -1)
            scores = y_pred.view(1, -1)
        else:  # Multiclass
            num_classes = y_pred.size(1)
            if y.max() + 1 > num_classes:
                raise ValueError(
                    f"y_pred contains fewer classes than y. Number of classes in prediction is {num_classes}"
                    f" and an element in y has invalid class = {y.max().item() + 1}."
                )
            P = y.view(-1)
            scores = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)

        return scores, P

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
        scores, P = self._prepare_output(output)
        self._scores.append(scores.to(self._device))
        self._P.append(P.to(self._device, dtype=torch.uint8 if self._type != "multiclass" else torch.long))

    def _compute_recall_and_precision(
        self, TP: torch.Tensor, scores: torch.Tensor, P: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Measuring recall & precision.

        Shape of function inputs and return values follow the table below. C is the number of classes, 1 for binary
        data. N is the number of samples. Finally, \#unique scores represents number of unique scores in ``scores``
        which is actually the number of thresholds.

        =================== =======================================
        **Object**          **Shape**
        =================== =======================================
        TP                  (N,)
        scores              (N,)
        P                   () (A single float)
        recall              (\#unique scores,)
        precision           (\#unique scores,)
        =================== =======================================

        Returns:
            `(recall, precision)`
        """
        indices = torch.argsort(scores, dim=-1, stable=True, descending=True)
        tp_summation = TP[..., indices].cumsum(dim=-1).double()

        # Adopted from Scikit-learn's implementation
        unique_scores_indices = torch.nonzero(
            scores.take_along_dim(indices).diff(append=(scores.max() + 1).unsqueeze(dim=0)), as_tuple=True
        )[0]
        tp_summation = tp_summation[..., unique_scores_indices]
        fp_summation = (unique_scores_indices + 1) - tp_summation

        if P == 0:
            # To be aligned with Scikit-Learn
            recall = torch.ones_like(tp_summation, device=self._device, dtype=torch.float)
        else:
            recall = tp_summation / P

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

        rank_P = (
            torch.cat(self._P, dim=-1)
            if self._P
            else (
                torch.empty((num_classes, 0), dtype=torch.uint8, device=self._device)
                if self._type == "multilabel"
                else torch.tensor(
                    [], dtype=torch.long if self._type == "multiclass" else torch.uint8, device=self._device
                )
            )
        )
        rank_P_shapes = cast(torch.Tensor, idist.all_gather(torch.tensor(rank_P.shape))).view(-1, len(rank_P.shape))
        P = torch.cat(all_gather_tensors_with_shapes(rank_P, rank_P_shapes.tolist()), dim=-1)

        rank_scores = (
            torch.cat(self._scores, dim=-1)
            if self._scores
            else (
                torch.tensor([], device=self._device)
                if self._type == "binary"
                else torch.empty((num_classes, 0), dtype=torch.double, device=self._device)
            )
        )
        rank_scores_shapes = cast(torch.Tensor, idist.all_gather(torch.tensor(rank_scores.shape))).view(
            -1, len(rank_scores.shape)
        )
        scores = torch.cat(all_gather_tensors_with_shapes(rank_scores, rank_scores_shapes.tolist()), dim=-1)

        if self._type == "multiclass":
            P = to_onehot(P, num_classes=num_classes).T
        if self.class_mean == "micro":
            P = P.reshape(1, -1)
            scores = scores.view(1, -1)
        P_count = P.sum(dim=-1)
        average_precisions = torch.zeros_like(P_count, device=self._device, dtype=torch.double)
        for cls in range(len(P_count)):
            recall, precision = self._compute_recall_and_precision(P[cls], scores[cls], P_count[cls])
            average_precisions[cls] = self._compute_average_precision(recall, precision)
        if self._type == "binary":
            return average_precisions.item()
        if self.class_mean is None:
            return average_precisions
        elif self.class_mean == "weighted":
            return torch.sum(P_count * average_precisions) / P_count.sum()
        else:
            return average_precisions.mean()
