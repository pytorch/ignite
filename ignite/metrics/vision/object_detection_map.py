from typing import Callable, cast, Dict, Literal, Optional, Sequence, Tuple, Union

import torch

from ignite.metrics.mean_average_precision import MeanAveragePrecision


class ObjectDetectionMAP(MeanAveragePrecision):
    def __init__(
        self,
        iou_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        flavor: Optional[Literal["COCO",]] = "COCO",
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        r"""Calculate the mean average precision for evaluating an object detector.

        The input to metric's ``update`` method should be a binary tuple of str-to-tensor dictionaries, (y_pred, y),
        which their items are as follows. N\ :sub:`det` and N\ :sub:`gt` are number of detections and ground truths
        respectively.

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
        ========= ================== =================================================

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

        super(ObjectDetectionMAP, self).__init__(
            rec_thresholds=rec_thresholds,
            average="max-precision" if flavor == "COCO" else "precision",
            class_mean="with_other_dims",
            output_transform=output_transform,
            device=device,
        )

    def _compute_recall_and_precision(
        self, TP: torch.Tensor, FP: Union[torch.Tensor, None], scores: torch.Tensor, P: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Measuring recall & precision

        This method is overriden since in the pycocotools reference implementation, when there are predictions with the
        same scores, they're considered associated with different thresholds in the course of measuring recall
        values, although it's not logically correct as those predictions are really associated with a single threshold,
        thus outputing a single recall value.
        """
        indices = torch.argsort(scores, dim=-1, stable=True, descending=True)
        tp = TP[..., indices]
        tp_summation = tp.cumsum(dim=-1).double()
        fp = cast(torch.Tensor, FP)[..., indices]
        fp_summation = fp.cumsum(dim=-1).double()

        recall = tp_summation / P
        predicted_positive = tp_summation + fp_summation
        precision = tp_summation / torch.where(predicted_positive == 0, 1, predicted_positive)

        return recall, precision

    def _measure_average_precision(self, recall: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
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
            return super()._measure_average_precision(recall, precision)

        precision_integrand = (
            precision.flip(-1).cummax(dim=-1).values.flip(-1) if self.average == "max-precision" else precision
        )
        rec_thresholds = cast(torch.Tensor, self.rec_thresholds).repeat((*recall.shape[:-1], 1))
        rec_thresh_indices = torch.searchsorted(recall, rec_thresholds)
        precision_integrand = precision_integrand.take_along_dim(
            rec_thresh_indices.where(rec_thresh_indices != recall.size(-1), 0), dim=-1
        ).where(rec_thresh_indices != recall.size(-1), 0)
        return torch.sum(precision_integrand, dim=-1) / len(cast(torch.Tensor, self.rec_thresholds))

    def compute(self) -> Union[torch.Tensor, float]:
        if not sum(cast(Dict[int, int], self._P).values()) and self.flavor == "COCO":
            return -1
        return super().compute()

    def _do_matching(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], Dict[int, torch.Tensor]]:
        """
        Matching logic of object detection mAP.
        """
        labels = target["labels"].detach()
        pred_labels = pred["labels"].detach()
        pred_scores = pred["scores"].detach()
        categories = list(set(labels.int().tolist() + pred_labels.int().tolist()))

        pred_boxes = pred["bbox"]
        gt_boxes = target["bbox"]

        is_crowd = target["iscrowd"]

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
