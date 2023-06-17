from collections import defaultdict
from typing import Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch

import ignite.distributed as idist
from ignite.distributed.utils import all_gather_tensors_with_shapes
from ignite.metrics.mean_average_precision import _BaseMeanAveragePrecision

from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.recall import _BasePrecisionRecall


class ObjectDetectionMAP(_BaseMeanAveragePrecision):
    _tp: Dict[int, List[torch.Tensor]]
    _fp: Dict[int, List[torch.Tensor]]
    _scores: Dict[int, List[torch.Tensor]]
    _P: Dict[int, int]

    def __init__(
        self,
        iou_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        flavor: Optional["Literal['COCO']"] = "COCO",
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        r"""Calculate the mean average precision for evaluating an object detector.

        Args:
            iou_thresholds: sequence of IoU thresholds to be considered for computing mean average precision.
                Values should be between 0 and 1. If not given, it's determined by ``flavor`` argument.
            flavor: string values so that metric computation recipe correspond to its respective flavor. For now, only
                available option is 'COCO'. Default 'COCO'.
            rec_thresholds: sequence of recall thresholds to be considered for computing mean average precision.
                Values should be between 0 and 1. If not given, it's determined by ``flavor`` argument.
            output_transform: a callable that is used to transform the
                :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
                By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
            device: specifies which device updates are accumulated on. Setting the
                metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
                non-blocking. By default, CPU.
        """
        try:
            from torchvision.ops.boxes import _box_inter_union, box_area

            def box_iou(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iscrowd: torch.BoolTensor) -> torch.Tensor:
                inter, union = _box_inter_union(pred_boxes, gt_boxes)
                union[:, iscrowd] = box_area(pred_boxes).reshape(-1, 1)
                iou = inter / union
                iou[iou.isnan()] = 0
                return iou

            self.box_iou = box_iou
        except ImportError:
            raise ModuleNotFoundError("This metric requires torchvision to be installed.")

        if flavor != "COCO":
            raise ValueError(f"Currently, the only available flavor for ObjectDetectionMAP is 'COCO', given {flavor}")
        self.flavor = flavor

        if iou_thresholds is None:
            iou_thresholds = torch.linspace(0.5, 0.95, 10, dtype=torch.double)

        self.iou_thresholds = self._setup_thresholds(iou_thresholds, "iou_thresholds")

        if rec_thresholds is None:
            rec_thresholds = torch.linspace(0, 1, 101, device=device, dtype=torch.double)

        super().__init__(
            rec_thresholds=rec_thresholds,
            average="max-precision" if flavor == "COCO" else "precision",
            class_mean="with_other_dims",
            output_transform=output_transform,
            device=device,
        )

    @reinit__is_reduced
    def reset(self) -> None:
        """
        Reset method of the metric
        """
        super(_BasePrecisionRecall, self).reset()

        self._tp = defaultdict(lambda: [])
        self._fp = defaultdict(lambda: [])
        self._scores = defaultdict(lambda: [])
        self._P = defaultdict(lambda: 0)
        self._num_classes: int = 0

    def _check_matching_input(self, output: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        y_pred_keys = {"bbox", "scores", "labels"}
        if (output[0].keys() & y_pred_keys) != y_pred_keys:
            raise ValueError(
                "y_pred dict in update's input should have 'bbox', 'scores'"
                f" and 'labels' keys. It has {output[0].keys()}"
            )

        y_keys = {"bbox", "labels"}
        if (output[1].keys() & y_keys) != y_keys:
            raise ValueError(
                "y dict in update's input should have 'bbox', 'labels'"
                f" and optionaly 'iscrowd' keys. It has {output[1].keys()}"
            )

    def _compute_recall_and_precision(
        self, TP: torch.Tensor, FP: torch.Tensor, scores: torch.Tensor, P: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Measuring recall & precision

        This method is overriden since in the pycocotools reference implementation, when there are predictions with the
        same scores, they're considered associated with different thresholds in the course of measuring recall
        values, although it's not logically correct as those predictions are really associated with a single threshold,
        thus outputing a single recall value.

        Shape of function inputs and return values follow the table below. N\ :sub:`pred` is the number of detections
        or predictions. ``...`` stands for the possible additional dimensions. Finally, \#unique scores represents
        number of unique scores in ``scores`` which is actually the number of thresholds.

        ============== ======================
        **Object**     **Shape**
        ============== ======================
        TP and FP      (..., N\ :sub:`pred`)
        scores         (N\ :sub:`pred`,)
        P              () (A single float,
                       greater than zero)
        recall         (..., \#unique scores)
        precision      (..., \#unique scores)
        ============== ======================

        Returns:
            `(recall, precision)`
        """
        indices = torch.argsort(scores, dim=-1, stable=True, descending=True)
        tp = TP[..., indices]
        tp_summation = tp.cumsum(dim=-1).double()
        fp = FP[..., indices]
        fp_summation = fp.cumsum(dim=-1).double()

        recall = tp_summation / P
        predicted_positive = tp_summation + fp_summation
        precision = tp_summation / torch.where(predicted_positive == 0, 1, predicted_positive)

        return recall, precision

    def _compute_average_precision(self, recall: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        """Measuring average precision.
        This method is overriden since :math:`1/#recall_thresholds` is used instead of :math:`r_k - r_{k-1}`
        as the recall differential in COCO flavor.

        Args:
            recall: n-dimensional tensor whose last dimension is the dimension of the samples. Should be ordered in
                ascending order in its last dimension.
            precision: like ``recall`` in the shape.

        Returns:
            average_precision: (n-1)-dimensional tensor containing the average precision for mean dimensions.
        """
        if self.flavor != "COCO":
            return super()._compute_average_precision(recall, precision)

        precision_integrand = (
            precision.flip(-1).cummax(dim=-1).values.flip(-1) if self.average == "max-precision" else precision
        )
        rec_thresholds = cast(torch.Tensor, self.rec_thresholds).repeat((*recall.shape[:-1], 1))
        rec_thresh_indices = torch.searchsorted(recall, rec_thresholds)
        precision_integrand = precision_integrand.take_along_dim(
            rec_thresh_indices.where(rec_thresh_indices != recall.size(-1), 0), dim=-1
        ).where(rec_thresh_indices != recall.size(-1), 0)
        return torch.sum(precision_integrand, dim=-1) / len(cast(torch.Tensor, self.rec_thresholds))

    def _do_matching(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], Dict[int, torch.Tensor]]:
        r"""
        Matching logic of object detection mAP.

        The method returns a quadrople of dictionaries containing TP, FP, P (actual positive) counts and scores for
        each class respectively. Please note that class numbers start from zero.

        Values in TP and FP are (m+1)-dimensional tensors of type ``uint8`` and shape
        (D\ :sub:`1`, D\ :sub:`2`, ..., D\ :sub:`m`, n\ :sub:`cls`) in which D\ :sub:`i`\ 's are possible additional
        dimensions (excluding the class dimension) mean of the average precision is taken over. n\ :sub:`cls` is the
        number of predictions for class `cls` which is the same for TP and FP.

        Note:
            TP and FP values are stored as uint8 tensors internally to avoid bool-to-uint8 copies before collective
            operations, as PyTorch colective operations `do not <https://github.com/pytorch/pytorch/issues/89197>`_
            support boolean tensors, at least on Gloo backend.

        P counts contains the number of ground truth samples for each class. Finally, the values in scores are 1-dim
        tensors of shape (n\ :sub:`cls`,) containing score or confidence of the predictions (doesn't need to be in
        [0,1]). If there is no prediction or ground truth for a class, it is absent from (TP, FP, scores) and P
        dictionaries respectively.

        Args:
            pred: First member of :meth:`update`'s input is given as this argument.
            target: Second member of :meth:`update`'s input is given as this argument.

        Returns:
            `(TP, FP, P, scores)` A quadrople of true positives, false positives, number of actual positives and scores.
        """
        labels = target["labels"].detach()
        pred_labels = pred["labels"].detach()
        pred_scores = pred["scores"].detach()
        categories = list(set(labels.int().tolist() + pred_labels.int().tolist()))

        pred_boxes = pred["bbox"]
        gt_boxes = target["bbox"]

        is_crowd = target["iscrowd"] if "iscrowd" in target else torch.zeros_like(target["labels"], dtype=torch.bool)

        tp: Dict[int, torch.Tensor] = {}
        fp: Dict[int, torch.Tensor] = {}
        P: Dict[int, int] = {}
        scores: Dict[int, torch.Tensor] = {}

        for category in categories:
            class_index_gt = labels == category
            num_category_gt = class_index_gt.sum()
            category_is_crowd = is_crowd[class_index_gt]
            if num_category_gt:
                P[category] = num_category_gt - category_is_crowd.sum()

            class_index_dt = pred_labels == category
            if not class_index_dt.any():
                continue

            scores[category] = pred_scores[class_index_dt]

            category_tp = torch.zeros(
                (len(self.iou_thresholds), class_index_dt.sum().item()), dtype=torch.uint8, device=self._device
            )
            category_fp = torch.zeros(
                (len(self.iou_thresholds), class_index_dt.sum().item()), dtype=torch.uint8, device=self._device
            )
            if num_category_gt:
                class_iou = self.box_iou(
                    pred_boxes[class_index_dt],
                    gt_boxes[class_index_gt],
                    cast(torch.BoolTensor, category_is_crowd.bool()),
                )
                class_maximum_iou = class_iou.max()
                category_pred_idx_sorted_by_decreasing_score = torch.argsort(
                    pred_scores[class_index_dt], stable=True, descending=True
                ).tolist()
                for thres_idx, iou_thres in enumerate(self.iou_thresholds):
                    if iou_thres <= class_maximum_iou:
                        matched_gt_indices = set()
                        for pred_idx in category_pred_idx_sorted_by_decreasing_score:
                            match_iou, match_idx = min(iou_thres, 1 - 1e-10), -1
                            for gt_idx in range(num_category_gt):
                                if (class_iou[pred_idx][gt_idx] < iou_thres) or (
                                    gt_idx in matched_gt_indices and torch.logical_not(category_is_crowd[gt_idx])
                                ):
                                    continue
                                if match_idx == -1 or (
                                    class_iou[pred_idx][gt_idx] >= match_iou
                                    and torch.logical_or(
                                        torch.logical_not(category_is_crowd[gt_idx]), category_is_crowd[match_idx]
                                    )
                                ):
                                    match_iou = class_iou[pred_idx][gt_idx]
                                    match_idx = gt_idx
                            if match_idx != -1:
                                matched_gt_indices.add(match_idx)
                                category_tp[thres_idx][pred_idx] = torch.logical_not(category_is_crowd[match_idx])
                            else:
                                category_fp[thres_idx][pred_idx] = 1
                    else:
                        category_fp[thres_idx] = 1
            else:
                category_fp[:, :] = 1

            tp[category] = category_tp
            fp[category] = category_fp

        return tp, fp, P, scores

    @reinit__is_reduced
    def update(self, output: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]) -> None:
        r"""Metric update function using prediction and target.

        Args:
            output: a binary tuple of str-to-tensor dictionaries, (y_pred, y), which their items
                are as follows. N\ :sub:`det` and N\ :sub:`gt` are number of detections and
                ground truths respectively.

                =======   ================== =================================================
                **y_pred items**
                ------------------------------------------------------------------------------
                Key       Value shape        Description
                =======   ================== =================================================
                'bbox'    (N\ :sub:`det`, 4) Bounding boxes of form (x1, y1, x2, y2)
                                             containing top left and bottom right coordinates.
                'score'   (N\ :sub:`det`,)   Confidence score of detections.
                'label'   (N\ :sub:`det`,)   Predicted category number of detections.
                =======   ================== =================================================

                ========= ================== =================================================
                **y items**
                ------------------------------------------------------------------------------
                Key       Value shape        Description
                ========= ================== =================================================
                'bbox'    (N\ :sub:`gt`, 4)  Bounding boxes of form (x1, y1, x2, y2)
                                             containing top left and bottom right coordinates.
                'label'   (N\ :sub:`gt`,)    Category number of ground truths.
                'iscrowd' (N\ :sub:`gt`,)    Whether ground truth boxes are crowd ones or not.
                                             It's optional with default value of ``False``.
                ========= ================== =================================================
        """
        self._check_matching_input(output)
        tps, fps, ps, scores_dict = self._do_matching(output[0], output[1])
        for cls in tps:
            self._tp[cls].append(tps[cls].to(device=self._device, dtype=torch.uint8))
            self._fp[cls].append(fps[cls].to(device=self._device, dtype=torch.uint8))
            self._scores[cls].append(scores_dict[cls].to(self._device))
        for cls in ps:
            self._P[cls] += ps[cls]
        classes = tps.keys() | ps.keys()
        if classes:
            self._num_classes = max(max(classes) + 1, self._num_classes)

    def compute(self) -> Union[torch.Tensor, float]:
        """
        Compute method of the metric
        """
        if sum(self._P.values()) < 1 and self.flavor == "COCO":
            return -1

        num_classes = int(idist.all_reduce(self._num_classes or 0, "MAX"))
        if num_classes < 1:
            return 0.0

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
            return 0.0
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

        average_precisions = -torch.ones(
            (num_classes, *mean_dimensions_shape),
            device=self._device,
            dtype=torch.double,
        )
        for cls in range(num_classes):
            if P[cls] == 0:
                continue

            num_preds_across_ranks = num_preds_per_class_across_ranks[:, [cls]]
            if num_preds_across_ranks.sum() == 0:
                average_precisions[cls] = 0
                continue
            shape_across_ranks = [
                (*mean_dimensions_shape, num_pred_in_rank.item()) for num_pred_in_rank in num_preds_across_ranks
            ]
            TP = torch.cat(
                all_gather_tensors_with_shapes(
                    torch.cat(self._tp[cls], dim=-1)
                    if self._tp[cls]
                    else torch.empty((*mean_dimensions_shape, 0), dtype=torch.uint8, device=self._device),
                    shape_across_ranks,
                ),
                dim=-1,
            )
            FP = torch.cat(
                all_gather_tensors_with_shapes(
                    torch.cat(self._fp[cls], dim=-1)
                    if self._fp[cls]
                    else torch.empty((*mean_dimensions_shape, 0), dtype=torch.uint8, device=self._device),
                    shape_across_ranks,
                ),
                dim=-1,
            )
            scores = torch.cat(
                all_gather_tensors_with_shapes(
                    torch.cat(self._scores[cls])
                    if self._scores[cls]
                    else torch.tensor([], dtype=torch.double, device=self._device),
                    num_preds_across_ranks.tolist(),
                )
            )
            recall, precision = self._compute_recall_and_precision(TP, FP, scores, P[cls])
            average_precision_for_cls_across_other_dims = self._compute_average_precision(recall, precision)
            average_precisions[cls] = average_precision_for_cls_across_other_dims

        return average_precisions[average_precisions > -1].mean()
