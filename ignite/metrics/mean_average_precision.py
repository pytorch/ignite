import itertools
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
from typing_extensions import Literal

import ignite.distributed as idist
from ignite.distributed.utils import _all_gather_tensors_with_shapes
from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.recall import _BasePrecisionRecall
from ignite.utils import to_onehot


class MeanAveragePrecision(_BasePrecisionRecall):
    _tp: Dict[int, List[torch.Tensor]]
    _fp: Dict[int, List[torch.Tensor]]
    _scores: Union[Dict[int, List[torch.Tensor]], List[torch.Tensor]]
    _P: Union[Dict[int, int], List[torch.Tensor]]

    def __init__(
        self,
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        average: Optional[Literal["precision", "max-precision"]] = "precision",
        class_mean: Optional[Literal["micro", "macro", "weighted", "with_other_dims"]] = "macro",
        classification_is_multilabel: bool = False,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        r"""Calculate the mean average precision metric i.e. mean of the averaged-over-recall precision for detection
        and classification tasks.

        Mean average precision attempts to give a measure of detector or classifier precision at various
        sensivity levels a.k.a recall thresholds. This is done by summing precisions at different recall
        thresholds weighted by the change in recall, as if the area under precision-recall curve is being computed.
        Mean average precision is the computed by taking the mean of this average precision over different classes
        and possibly some additional dimensions in the detection task.

        For detection tasks user should use downstream metrics like
        :class:`~ignite.metrics.vision.object_detection_map.ObjectDetectionMAP` or subclass this metric and implement
        its :meth:`_do_matching` method to provide the metric with desired matching logic. Then this method is called
        internally in :meth:`update` method on prediction-target pairs. For classification, all the binary, multiclass
        and multilabel data are supported. In the latter case, ``classification_is_multilabel`` should be set to true.

        `mean` in the mean average precision accounts for mean of the average precision across classes. ``class_mean``
        determines how to take this mean. In the detection tasks, it's possible to take mean of the average precision
        in other respects as well e.g. IoU threshold in an object detection task. To this end, average precision
        corresponding to each value of IoU thresholds should get measured in :meth:`_do_matching`. Please refer to
        :meth:`_do_matching` for more info on this.

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

                  For multiclass inputs, this is equivalent with mean average accuracy.

                'weighted'
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

            classification_is_multilabel: Used in classification task and determines if the data
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

        super(_BasePrecisionRecall, self).__init__(
            output_transform=output_transform, is_multilabel=classification_is_multilabel, device=device
        )

        if self._task == "classification" and self.class_mean == "with_other_dims":
            raise ValueError("class_mean 'with_other_dims' is not compatible with classification.")

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

    @reinit__is_reduced
    def reset(self) -> None:
        """
        Reset method of the metric
        """
        super(_BasePrecisionRecall, self).reset()
        if self._do_matching.__func__ == MeanAveragePrecision._do_matching:  # type: ignore[attr-defined]
            self._task: Literal["classification", "detection"] = "classification"
        else:
            self._task = "detection"
        self._tp = defaultdict(lambda: [])
        self._fp = defaultdict(lambda: [])
        if self._task == "detection":
            self._scores = defaultdict(lambda: [])
            self._P = defaultdict(lambda: 0)
            self._num_classes = 0
        else:
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

    def _check_matching_output_shape(
        self, tps: Dict[int, torch.Tensor], fps: Dict[int, torch.Tensor], scores: Dict[int, torch.Tensor]
    ) -> None:
        if not (tps.keys() == fps.keys() == scores.keys()):
            raise ValueError(
                "Returned TP, FP and scores dictionaries from _do_matching should have"
                f" the same keys (classes), given {tps.keys()}, {fps.keys()} and {scores.keys()}"
            )
        try:
            cls = list(tps.keys()).pop()
        except IndexError:  # No prediction
            pass
        else:
            if tps[cls].dtype not in (torch.bool, torch.uint8):
                raise TypeError(f"Tensors in TP and FP dictionaries should be boolean or uint8, given {tps[cls].dtype}")

            if tps[cls].size(-1) != fps[cls].size(-1) != scores[cls].size(0):
                raise ValueError(
                    "Sample dimension of tensors in TP, FP and scores should have equal size per class,"
                    f"given {tps[cls].size(-1)}, {fps[cls].size(-1)} and {scores[cls].size(-1)} for class {cls}"
                    " respectively."
                )
        for self_tp_or_fp, new_tp_or_fp, name in [(self._tp, tps, "TP"), (self._fp, fps, "FP")]:
            new_tp_or_fp.keys()
            try:
                cls = (self_tp_or_fp.keys() & new_tp_or_fp.keys()).pop()
            except KeyError:
                pass
            else:
                if self_tp_or_fp[cls][-1].shape[:-1] != new_tp_or_fp[cls].shape[:-1]:
                    raise ValueError(
                        f"Tensors in returned {name} from _do_matching should not change in shape "
                        "except possibly in the last dimension which is the dimension of samples. Given "
                        f"{self_tp_or_fp[cls][-1].shape} and {new_tp_or_fp[cls].shape}"
                    )

    def _classification_prepare_output(
        self, y_pred: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _do_matching(
        self, pred: Any, target: Any
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], Dict[int, torch.Tensor]]:
        r"""
        Matching logic holder of the metric for detection tasks.

        The developer must implement this method by subclassing the metric. There is no constraint on type and shape of
        ``pred`` and ``target``, but the method should return a quadrople of dictionaries containing TP, FP,
        P (actual positive) counts and scores for each class respectively. Please note that class numbers start from
        zero.

        Values in TP and FP are (m+1)-dimensional tensors of type ``bool`` or ``uint8`` and shape
        (D\ :sub:`1`, D\ :sub:`2`, ..., D\ :sub:`m`, n\ :sub:`cls`) in which D\ :sub:`i`\ 's are possible additional
        dimensions (excluding the class dimension) mean of the average precision is taken over. n\ :sub:`cls` is the
        number of predictions for class `cls` which is the same for TP and FP.

        Note:
            TP and FP values are stored as uint8 tensors internally to avoid bool-to-uint8 copies before collective
            operations, as PyTorch colective operations `do not <https://github.com/pytorch/pytorch/issues/89197>`_
            support boolean tensors, at least on Gloo backend.


        P counts contains the number of ground truth samples for each class. Finally, the values in scores are 1-dim
        tensors of shape (n\ :sub:`cls`,) containing score or confidence of the predictions (doesn't need to be in
        [0,1]). If there is no prediction or ground truth for a class, it could be absent from (TP, FP, scores) and P
        dictionaries respectively.

        Args:
            pred: First member of :meth:`update`'s input is given as this argument. There's no constraint on its type
                and shape.
            target: Second member of :meth:`update`'s input is given as this argument. There's no constraint on its type
                and shape.

        Returns:
            `(TP, FP, P, scores)` A quadrople of true positives, false positives, number of actual positives and scores.
        """
        raise NotImplementedError(
            "Please subclass MeanAveragePrecision and implement `_do_matching` method"
            " to use the metric in detection."
        )

    @reinit__is_reduced
    def update(self, output: Union[Tuple[Any, Any], Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Metric update function using prediction and target.

        Args:
            output: a binary tuple. It should consist of prediction and target tensors in the classification case but
                for detection it is the same as the implemented-by-user :meth:`_do_matching`.

                For classification, this metric follows the same rules on ``output`` members shape as the
                :meth:`Precision.update <precision.Precision.update>` except for ``y_pred`` of binary and multilabel
                data which should be comprised of positive class probabilities here.
        """

        if self._task == "classification":
            self._check_shape(output)
            prediction, target = output[0].detach(), output[1].detach()
            self._check_type((prediction, target))
            scores, P = self._classification_prepare_output(prediction, target)
            cast(List[torch.Tensor], self._scores).append(scores.to(self._device))
            cast(List[torch.Tensor], self._P).append(
                P.to(self._device, dtype=torch.uint8 if self._type != "multiclass" else torch.long)
            )
        else:
            tps, fps, ps, scores_dict = self._do_matching(output[0], output[1])
            self._check_matching_output_shape(tps, fps, scores_dict)
            for cls in tps:
                self._tp[cls].append(tps[cls].to(device=self._device, dtype=torch.uint8))
                self._fp[cls].append(fps[cls].to(device=self._device, dtype=torch.uint8))
                cast(Dict[int, List[torch.Tensor]], self._scores)[cls].append(scores_dict[cls].to(self._device))
            for cls in ps:
                cast(Dict[int, int], self._P)[cls] += ps[cls]
            classes = tps.keys() | ps.keys()
            if classes:
                self._num_classes = max(max(classes) + 1, self._num_classes)

    def _compute_recall_and_precision(
        self, TP: torch.Tensor, FP: Union[torch.Tensor, None], scores: torch.Tensor, P: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Measuring recall & precision which is the common operation among different settings of the metric.

        Shape of function inputs and return values follow the table below. C is the number of classes, 1 for binary
        data. N\ :sub:`pred` is the number of detections or predictions which is the same as the number of samples in
        classification task. ``...`` stands for the additional dimensions in the detection task. Finally,
        \#unique scores represents number of unique scores in ``scores`` which is actually the number of thresholds.

        ============== ======================
        Detection task
        -------------------------------------
        **Object**     **Shape**
        ============== ======================
        TP and FP      (..., N\ :sub:`pred`)
        scores         (N\ :sub:`pred`,)
        P              () (A single float)
        recall         (..., \#unique scores)
        precision      (..., \#unique scores)
        ============== ======================

        =================== =======================================
        Classification task
        -----------------------------------------------------------
        **Object**          **Shape**
        =================== =======================================
        TP                  (N\ :sub:`pred`,)
        FP                  None (FP is computed here to be faster)
        scores              (N\ :sub:`pred`,)
        P                   () (A single float)
        recall              (\#unique scores,)
        precision           (\#unique scores,)
        =================== =======================================

        Returns:
            `(recall, precision)`
        """
        indices = torch.argsort(scores, dim=-1, stable=True, descending=True)
        tp = TP.take_along_dim(indices, dim=-1) if self._task == "classification" else TP[..., indices]
        tp_summation = tp.cumsum(dim=-1).double()

        # Adopted from Scikit-learn's implementation
        unique_scores_indices = torch.nonzero(
            scores.take_along_dim(indices).diff(append=(scores.max() + 1).unsqueeze(dim=0)), as_tuple=True
        )[0]
        tp_summation = tp_summation[..., unique_scores_indices]
        if self._task == "classification":
            fp_summation = (unique_scores_indices + 1) - tp_summation
        else:
            fp = cast(torch.Tensor, FP)[..., indices]
            fp_summation = fp.cumsum(dim=-1).double()
            fp_summation = fp_summation[..., unique_scores_indices]

        if self._task == "classification" and P == 0:
            recall = torch.ones_like(tp_summation, device=self._device, dtype=torch.bool)
        else:
            recall = tp_summation / P

        predicted_positive = tp_summation + fp_summation
        precision = tp_summation / torch.where(predicted_positive == 0, 1, predicted_positive)
        return recall, precision

    def _measure_average_precision(self, recall: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
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

    def compute(self) -> Union[torch.Tensor, float]:
        """
        Compute method of the metric
        """
        num_classes = int(idist.all_reduce(self._num_classes or 0, "MAX"))
        if not num_classes:
            return 0.0

        if self._task == "detection":
            P = cast(
                torch.Tensor,
                idist.all_reduce(torch.tensor(list(map(self._P.__getitem__, range(num_classes))), device=self._device)),
            )
            num_preds = torch.tensor(
                [sum([tp.shape[-1] for tp in self._tp[cls]]) if self._tp[cls] else 0 for cls in range(num_classes)],
                device=self._device,
            )
            num_preds_per_class_across_ranks = torch.stack(
                cast(torch.Tensor, idist.all_gather(num_preds)).split(split_size=num_classes)
            )
            if num_preds_per_class_across_ranks.sum() == 0:
                return (
                    0.0
                    if self.class_mean is not None
                    else torch.zeros((num_classes,), dtype=torch.double, device=self._device)
                )
            a_nonempty_rank, its_class_with_pred = list(zip(*torch.where(num_preds_per_class_across_ranks != 0))).pop(0)
            a_nonempty_rank = a_nonempty_rank.item()
            its_class_with_pred = its_class_with_pred.item()
            mean_dimensions_shape = cast(
                torch.Tensor,
                idist.broadcast(
                    torch.tensor(self._tp[its_class_with_pred][-1].shape[:-1], device=self._device)
                    if idist.get_rank() == a_nonempty_rank
                    else None,
                    a_nonempty_rank,
                    safe_mode=True,
                ),
            ).tolist()

            if self.class_mean != "micro":
                shapes_across_ranks = {
                    cls: [
                        (*mean_dimensions_shape, num_pred_in_rank)
                        for num_pred_in_rank in num_preds_per_class_across_ranks[:, cls]
                    ]
                    for cls in range(num_classes)
                }
                TP = {
                    cls: torch.cat(
                        _all_gather_tensors_with_shapes(
                            torch.cat(self._tp[cls], dim=-1)
                            if self._tp[cls]
                            else torch.empty((*mean_dimensions_shape, 0), dtype=torch.uint8, device=self._device),
                            shapes_across_ranks[cls],
                        ),
                        dim=-1,
                    )
                    for cls in range(num_classes)
                }
                FP = {
                    cls: torch.cat(
                        _all_gather_tensors_with_shapes(
                            torch.cat(self._fp[cls], dim=-1)
                            if self._fp[cls]
                            else torch.empty((*mean_dimensions_shape, 0), dtype=torch.uint8, device=self._device),
                            shapes_across_ranks[cls],
                        ),
                        dim=-1,
                    )
                    for cls in range(num_classes)
                }
                scores = {
                    cls: torch.cat(
                        _all_gather_tensors_with_shapes(
                            torch.cat(cast(List[torch.Tensor], self._scores[cls]))
                            if self._scores[cls]
                            else torch.tensor([], dtype=torch.double, device=self._device),
                            num_preds_per_class_across_ranks[:, [cls]].tolist(),
                        )
                    )
                    for cls in range(num_classes)
                }

                average_precisions = -torch.ones(
                    (num_classes, *(mean_dimensions_shape if self.class_mean == "with_other_dims" else ())),
                    device=self._device,
                    dtype=torch.double,
                )
                for cls in range(num_classes):
                    if P[cls] == 0:
                        continue
                    if TP[cls].size(-1) == 0:
                        average_precisions[cls] = 0
                        continue
                    recall, precision = self._compute_recall_and_precision(TP[cls], FP[cls], scores[cls], P[cls])
                    average_precision_for_cls_across_other_dims = self._measure_average_precision(recall, precision)
                    if self.class_mean != "with_other_dims":
                        average_precisions[cls] = average_precision_for_cls_across_other_dims.mean()
                    else:
                        average_precisions[cls] = average_precision_for_cls_across_other_dims
                if self.class_mean is None:
                    average_precisions[average_precisions == -1] = 0
                    return average_precisions
                elif self.class_mean == "weighted":
                    return torch.dot(P.double(), average_precisions) / P.sum()
                else:
                    return average_precisions[average_precisions > -1].mean()
            else:
                num_preds_across_ranks = num_preds_per_class_across_ranks.sum(dim=1)
                shapes_across_ranks_in_micro = [
                    (*mean_dimensions_shape, num_preds_in_rank.item()) for num_preds_in_rank in num_preds_across_ranks
                ]
                TP_micro = torch.cat(
                    _all_gather_tensors_with_shapes(
                        torch.cat(list(itertools.chain(*map(self._tp.__getitem__, range(num_classes)))), dim=-1).to(
                            torch.uint8
                        )
                        if num_preds_across_ranks[idist.get_rank()]
                        else torch.empty((*mean_dimensions_shape, 0), dtype=torch.uint8, device=self._device),
                        shapes_across_ranks_in_micro,
                    ),
                    dim=-1,
                ).bool()
                FP_micro = torch.cat(
                    _all_gather_tensors_with_shapes(
                        torch.cat(list(itertools.chain(*map(self._fp.__getitem__, range(num_classes)))), dim=-1).to(
                            torch.uint8
                        )
                        if num_preds_across_ranks[idist.get_rank()]
                        else torch.empty((*mean_dimensions_shape, 0), dtype=torch.uint8, device=self._device),
                        shapes_across_ranks_in_micro,
                    ),
                    dim=-1,
                ).bool()
                scores_micro = torch.cat(
                    _all_gather_tensors_with_shapes(
                        torch.cat(
                            list(
                                itertools.chain(
                                    *map(
                                        cast(Dict[int, List[torch.Tensor]], self._scores).__getitem__,
                                        range(num_classes),
                                    )
                                )
                            )
                        )
                        if num_preds_across_ranks[idist.get_rank()]
                        else torch.tensor([], dtype=torch.double, device=self._device),
                        num_preds_across_ranks.unsqueeze(dim=-1).tolist(),
                    )
                )
                P = P.sum()
                recall, precision = self._compute_recall_and_precision(TP_micro, FP_micro, scores_micro, P)
                return self._measure_average_precision(recall, precision).mean()
        else:
            rank_P = (
                torch.cat(cast(List[torch.Tensor], self._P), dim=-1)
                if self._P
                else (
                    torch.empty((num_classes, 0), dtype=torch.uint8, device=self._device)
                    if self._type == "multilabel"
                    else torch.tensor(
                        [], dtype=torch.long if self._type == "multiclass" else torch.uint8, device=self._device
                    )
                )
            )
            P = torch.cat(cast(List[torch.Tensor], idist.all_gather(rank_P, tensor_different_shape=True)), dim=-1)
            scores_classification = torch.cat(
                cast(
                    List[torch.Tensor],
                    idist.all_gather(
                        torch.cat(cast(List[torch.Tensor], self._scores), dim=-1)
                        if self._scores
                        else (
                            torch.tensor([], device=self._device)
                            if self._type == "binary"
                            else torch.empty((num_classes, 0), dtype=torch.double, device=self._device)
                        ),
                        tensor_different_shape=True,
                    ),
                ),
                dim=-1,
            )
            if self._type == "multiclass":
                P = to_onehot(P, num_classes=self._num_classes).T
            if self.class_mean == "micro":
                P = P.reshape(1, -1)
                scores_classification = scores_classification.view(1, -1)
            P_count = P.sum(dim=-1)
            average_precisions = torch.zeros_like(P_count, device=self._device, dtype=torch.double)
            for cls in range(len(P_count)):
                recall, precision = self._compute_recall_and_precision(
                    P[cls], None, scores_classification[cls], P_count[cls]
                )
                average_precisions[cls] = self._measure_average_precision(recall, precision)
            if self._type == "binary":
                return average_precisions.item()
            if self.class_mean is None:
                return average_precisions
            elif self.class_mean == "weighted":
                return torch.sum(P_count * average_precisions) / P_count.sum()
            else:
                return average_precisions.mean()
