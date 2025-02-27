from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
from typing_extensions import Literal

from ignite.metrics import MetricGroup

from ignite.metrics.mean_average_precision import _BaseAveragePrecision, _cat_and_agg_tensors
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


def coco_tensor_list_to_dict_list(
    output: Tuple[
        Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]],
        Union[List[torch.Tensor], List[Dict[str, torch.Tensor]]],
    ]
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """Convert either of output's `y_pred` or `y` from list of `(N, 6)` tensors to list of str-to-tensor dictionaries,
    or keep them unchanged if they're already in the deisred format.

    Input format is a `(N, 6)` or (`N, 5)` tensor which `N` is the number of predicted/target bounding boxes for the
    image and the second dimension contains `(x1, y1, x2, y2, confidence, class)`/`(x1, y1, x2, y2, class[, iscrowd])`.
    Output format is a str-to-tensor dictionary containing 'bbox' and `class` keys, plus `confidence` key for `y_pred`
    and possibly `iscrowd` for `y`.

    Args:
        output: `(y_pred,y)` tuple whose members are either list of tensors or list of dicts.

    Returns:
        `(y_pred,y)` tuple whose members are list of str-to-tensor dictionaries.
    """
    y_pred, y = output
    if len(y_pred) > 0 and isinstance(y_pred[0], torch.Tensor):
        y_pred = [{"bbox": t[:, :4], "confidence": t[:, 4], "class": t[:, 5]} for t in cast(List[torch.Tensor], y_pred)]
    if len(y) > 0 and isinstance(y[0], torch.Tensor):
        if y[0].size(1) == 5:
            y = [{"bbox": t[:, :4], "class": t[:, 4]} for t in cast(List[torch.Tensor], y)]
        else:
            y = [{"bbox": t[:, :4], "class": t[:, 4], "iscrowd": t[:, 5]} for t in cast(List[torch.Tensor], y)]
    return cast(Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]], (y_pred, y))


class ObjectDetectionAvgPrecisionRecall(Metric, _BaseAveragePrecision):
    _tps: List[torch.Tensor]
    _fps: List[torch.Tensor]
    _scores: List[torch.Tensor]
    _y_pred_labels: List[torch.Tensor]
    _y_true_count: torch.Tensor
    _num_classes: int

    def __init__(
        self,
        iou_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        num_classes: int = 80,
        max_detections_per_image_per_class: int = 100,
        area_range: Literal["small", "medium", "large", "all"] = "all",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        r"""Calculate mean average precision & recall for evaluating an object detector in the COCO way.


        Average precision is computed by averaging precision over increasing levels of recall thresholds.
        In COCO, the maximum precision across thresholds greater or equal a recall threshold is
        considered as the average summation operand; In other words, the precision peek across lower or equal
        sensivity levels is used for a recall threshold:

        .. math::
            \text{Average Precision} = \sum_{k=1}^{\#rec\_thresholds} (r_k - r_{k-1}) max(P_{k:})

        Average recall is the detector's maximum recall, considering all matched detections as TP,
        averaged over classes.

        Args:
            iou_thresholds: sequence of IoU thresholds to be considered for computing mean average precision & recall.
                Values should be between 0 and 1. If not given, COCO's default (.5, .55, ..., .95) would be used.
            rec_thresholds: sequence of recall thresholds to be considered for computing mean average precision.
                Values should be between 0 and 1. If not given, COCO's default (.0, .01, .02, ..., 1.) would be used.
            num_classes: number of categories. Default is 80, that of the COCO dataset.
            area_range: area range which only objects therein are considered in evaluation. By default, 'all'.
            max_detections_per_image_per_class: maximum number of detections per class in each image to consider
                for evaluation. The most confident ones are selected.
            output_transform: a callable that is used to transform the :class:`~ignite.engine.engine.Engine`'s
                ``process_function``'s output into the form expected by the metric. An already provided example is
                :func:`~ignite.metrics.vision.object_detection_average_precision_recall.coco_tensor_list_to_dict_list`
                which accepts `y_pred` and `y` as lists of tensors and transforms them to the expected format.
                Default is the identity function.
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

        if iou_thresholds is None:
            iou_thresholds = torch.linspace(0.5, 0.95, 10, dtype=torch.double)

        self._iou_thresholds = self._setup_thresholds(iou_thresholds, "iou_thresholds")

        if rec_thresholds is None:
            rec_thresholds = torch.linspace(0, 1, 101, dtype=torch.double)

        self._num_classes = num_classes
        self._area_range = area_range
        self._max_detections_per_image_per_class = max_detections_per_image_per_class

        super(ObjectDetectionAvgPrecisionRecall, self).__init__(
            output_transform=output_transform,
            device=device,
            skip_unrolling=skip_unrolling,
        )
        super(Metric, self).__init__(
            rec_thresholds=rec_thresholds,
            class_mean=None,
        )
        precision = torch.double if torch.device(device).type != "mps" else torch.float32
        self.rec_thresholds = cast(torch.Tensor, self.rec_thresholds).to(device=device, dtype=precision)

    @reinit__is_reduced
    def reset(self) -> None:
        self._tps = []
        self._fps = []
        self._scores = []
        self._y_pred_labels = []
        self._y_true_count = torch.zeros((self._num_classes,), device=self._device)

    def _match_area_range(self, bboxes: torch.Tensor) -> torch.Tensor:
        from torchvision.ops.boxes import box_area

        areas = box_area(bboxes)
        if self._area_range == "all":
            min_area = 0
            max_area = 1e10
        elif self._area_range == "small":
            min_area = 0
            max_area = 1024
        elif self._area_range == "medium":
            min_area = 1024
            max_area = 9216
        elif self._area_range == "large":
            min_area = 9216
            max_area = 1e10
        return torch.logical_and(areas >= min_area, areas <= max_area)

    def _check_matching_input(
        self, output: Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]
    ) -> None:
        y_pred, y = output
        if len(y_pred) != len(y):
            raise ValueError(f"y_pred and y should have the same number of samples, given {len(y_pred)} and {len(y)}.")
        if len(y_pred) == 0:
            raise ValueError("y_pred and y should contain at least one sample.")

        y_pred_keys = {"bbox", "scores", "labels"}
        if (y_pred[0].keys() & y_pred_keys) != y_pred_keys:
            raise ValueError(
                "y_pred sample dictionaries should have 'bbox', 'scores'"
                f" and 'labels' keys, given keys: {y_pred[0].keys()}"
            )

        y_keys = {"bbox", "labels"}
        if (y[0].keys() & y_keys) != y_keys:
            raise ValueError(
                "y sample dictionaries should have 'bbox', 'labels'"
                f" and optionally 'iscrowd' keys, given keys: {y[0].keys()}"
            )

    def _compute_recall_and_precision(
        self, TP: torch.Tensor, FP: torch.Tensor, scores: torch.Tensor, y_true_count: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Measuring recall & precision

        This method is different from that of MeanAveragePrecision since in the pycocotools reference implementation,
        when there are predictions with the same scores, they're considered associated with different thresholds in the
        course of measuring recall values, although it's not logically correct as those predictions are really
        associated with a single threshold, thus outputing a single recall value.

        Shape of function inputs and return values follow the table below. N\ :sub:`pred` is the number of detections
        or predictions. ``...`` stands for the possible additional dimensions. Finally, \#unique scores represents
        number of unique scores in ``scores`` which is actually the number of thresholds.

        ============== ======================
        **Object**     **Shape**
        ============== ======================
        TP and FP      (..., N\ :sub:`pred`)
        scores         (N\ :sub:`pred`,)
        y_true_count   () (A single float,
                       greater than zero)
        recall         (..., \#unique scores)
        precision      (..., \#unique scores)
        ============== ======================

        Returns:
            `(recall, precision)`
        """
        indices = torch.argsort(scores, dim=-1, stable=True, descending=True)
        tp = TP[..., indices]
        tp_summation = tp.cumsum(dim=-1)
        if tp_summation.device.type != "mps":
            tp_summation = tp_summation.double()

        fp = FP[..., indices]
        fp_summation = fp.cumsum(dim=-1)
        if fp_summation.device.type != "mps":
            fp_summation = fp_summation.double()

        recall = tp_summation / y_true_count
        predicted_positive = tp_summation + fp_summation
        precision = tp_summation / torch.where(predicted_positive == 0, 1, predicted_positive)

        return recall, precision

    def _compute_average_precision(self, recall: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
        """Measuring average precision.
        This method is overriden since :math:`1/#recall_thresholds` is used instead of :math:`r_k - r_{k-1}`
        as the recall differential in COCO's reference implementation i.e., pycocotools.

        Args:
            recall: n-dimensional tensor whose last dimension is the dimension of the samples. Should be ordered in
                ascending order in its last dimension.
            precision: like ``recall`` in the shape.

        Returns:
            average_precision: (n-1)-dimensional tensor containing the average precision for mean dimensions.
        """
        if precision.device.type == "mps":
            # Manual fallback to CPU if precision is on MPS due to the error:
            # NotImplementedError: The operator 'aten::_cummax_helper' is not currently implemented for the MPS device
            device = precision.device
            precision_integrand = precision.flip(-1).cpu()
            precision_integrand = precision_integrand.cummax(dim=-1).values
            precision_integrand = precision_integrand.to(device=device).flip(-1)
        else:
            precision_integrand = precision.flip(-1).cummax(dim=-1).values.flip(-1)
        rec_thresholds = cast(torch.Tensor, self.rec_thresholds).repeat((*recall.shape[:-1], 1))
        rec_thresh_indices = (
            torch.searchsorted(recall, rec_thresholds)
            if recall.size(-1) != 0
            else torch.LongTensor([], device=self._device)
        )
        precision_integrand = precision_integrand.take_along_dim(
            rec_thresh_indices.where(rec_thresh_indices != recall.size(-1), 0), dim=-1
        ).where(rec_thresh_indices != recall.size(-1), 0)
        return torch.sum(precision_integrand, dim=-1) / len(cast(torch.Tensor, self.rec_thresholds))

    @reinit__is_reduced
    def update(self, output: Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]) -> None:
        r"""Metric update method using prediction and target.

        Args:
            output: a tuple, (y_pred, y), of two same-length lists, each one containing
                str-to-tensor dictionaries whose items is as follows. N\ :sub:`det` and
                N\ :sub:`gt` are number of detections and ground truths for a sample
                respectively.

                ======== ================== =================================================
                **y_pred items**
                -----------------------------------------------------------------------------
                Key      Value shape        Description
                ======== ================== =================================================
                'bbox'   (N\ :sub:`det`, 4) Bounding boxes of form (x1, y1, x2, y2)
                                            containing top left and bottom right coordinates.
                'scores' (N\ :sub:`det`,)   Confidence score of detections.
                'labels' (N\ :sub:`det`,)   Predicted category number of detections in
                                            `torch.long` dtype.
                ======== ================== =================================================

                ========= ================= =================================================
                **y items**
                -----------------------------------------------------------------------------
                Key       Value shape        Description
                ========= ================= =================================================
                'bbox'    (N\ :sub:`gt`, 4) Bounding boxes of form (x1, y1, x2, y2)
                                            containing top left and bottom right coordinates.
                'labels'  (N\ :sub:`gt`,)   Category number of ground truths in `torch.long`
                                            dtype.
                'iscrowd' (N\ :sub:`gt`,)   Whether ground truth boxes are crowd ones or not.
                                            This key is optional.
                ========= ================= =================================================
        """
        self._check_matching_input(output)
        for pred, target in zip(*output):
            labels = target["labels"]
            gt_boxes = target["bbox"]
            gt_is_crowd = (
                target["iscrowd"].bool() if "iscrowd" in target else torch.zeros_like(labels, dtype=torch.bool)
            )
            gt_ignore = ~self._match_area_range(gt_boxes) | gt_is_crowd
            self._y_true_count += torch.bincount(labels[~gt_ignore], minlength=self._num_classes).to(
                device=self._device
            )

            # Matching logic of object detection mAP, according to COCO reference implementation.
            if len(pred["labels"]):
                best_detections_index = torch.argsort(pred["scores"], stable=True, descending=True)
                max_best_detections_index = torch.cat(
                    [
                        best_detections_index[pred["labels"][best_detections_index] == c][
                            : self._max_detections_per_image_per_class
                        ]
                        for c in range(self._num_classes)
                    ]
                )
                pred_boxes = pred["bbox"][max_best_detections_index]
                pred_labels = pred["labels"][max_best_detections_index]
                if not len(labels):
                    tp = torch.zeros(
                        (len(self._iou_thresholds), len(max_best_detections_index)),
                        dtype=torch.uint8,
                        device=self._device,
                    )
                    self._tps.append(tp)
                    self._fps.append(~tp & self._match_area_range(pred_boxes).to(self._device))
                else:
                    ious = self.box_iou(pred_boxes, gt_boxes, cast(torch.BoolTensor, gt_is_crowd))
                    category_no_match = labels.expand(len(pred_labels), -1) != pred_labels.view(-1, 1)
                    NO_MATCH = -3
                    ious[category_no_match] = NO_MATCH
                    ious = ious.unsqueeze(-1).repeat((1, 1, len(self._iou_thresholds)))
                    ious[ious < self._iou_thresholds] = NO_MATCH
                    IGNORANCE = -2
                    ious[:, gt_ignore] += IGNORANCE
                    for i in range(len(pred_labels)):
                        # Flip is done to give priority to the last item with maximal value,
                        # as torch.max selects the first one.
                        match_gts = ious[i].flip(0).max(0)
                        match_gts_indices = ious.size(1) - 1 - match_gts.indices
                        for t in range(len(self._iou_thresholds)):
                            if match_gts.values[t] > NO_MATCH and not gt_is_crowd[match_gts_indices[t]]:
                                ious[:, match_gts_indices[t], t] = NO_MATCH
                                ious[i, match_gts_indices[t], t] = match_gts.values[t]

                    max_ious = ious.max(1).values
                    self._tps.append((max_ious >= 0).T.to(dtype=torch.uint8, device=self._device))
                    self._fps.append(
                        ((max_ious <= NO_MATCH).T & self._match_area_range(pred_boxes)).to(
                            dtype=torch.uint8, device=self._device
                        )
                    )

                scores = pred["scores"][max_best_detections_index]
                if self._device.type == "mps" and scores.dtype == torch.double:
                    scores = scores.to(dtype=torch.float32)
                self._scores.append(scores.to(self._device))
                self._y_pred_labels.append(pred_labels.to(dtype=torch.int, device=self._device))

    @sync_all_reduce("_y_true_count")
    def _compute(self) -> torch.Tensor:
        pred_labels = _cat_and_agg_tensors(self._y_pred_labels, cast(Tuple[int], ()), torch.int, self._device)
        TP = _cat_and_agg_tensors(self._tps, (len(self._iou_thresholds),), torch.uint8, self._device)
        FP = _cat_and_agg_tensors(self._fps, (len(self._iou_thresholds),), torch.uint8, self._device)
        fp_precision = torch.double if self._device.type != "mps" else torch.float32
        scores = _cat_and_agg_tensors(self._scores, cast(Tuple[int], ()), fp_precision, self._device)

        average_precisions_recalls = -torch.ones(
            (2, self._num_classes, len(self._iou_thresholds)),
            device=self._device,
            dtype=fp_precision,
        )
        for cls in range(self._num_classes):
            if self._y_true_count[cls] == 0:
                continue

            cls_labels = pred_labels == cls
            if sum(cls_labels) == 0:
                average_precisions_recalls[:, cls] = 0.0
                continue

            recall, precision = self._compute_recall_and_precision(
                TP[..., cls_labels], FP[..., cls_labels], scores[cls_labels], self._y_true_count[cls]
            )
            average_precision_for_cls_per_iou_threshold = self._compute_average_precision(recall, precision)
            average_precisions_recalls[0, cls] = average_precision_for_cls_per_iou_threshold
            average_precisions_recalls[1, cls] = recall[..., -1]
        return average_precisions_recalls

    def compute(self) -> Tuple[float, float]:
        average_precisions_recalls = self._compute()
        if (average_precisions_recalls == -1).all():
            return -1.0, -1.0
        ap = average_precisions_recalls[0][average_precisions_recalls[0] > -1].mean().item()
        ar = average_precisions_recalls[1][average_precisions_recalls[1] > -1].mean().item()
        return ap, ar


class CommonObjectDetectionMetrics(MetricGroup):
    """
    Common Object Detection metrics. Included metrics are as follows:

    =============== ==========================================
    **Metric name**    **Description**
    =============== ==========================================
    AP@50..95       Average precision averaged over
                    .50 to.95 IOU thresholds
    AR-100          Average recall with maximum 100 detections
    AP@50           Average precision with IOU threshold=.50
    AP@75           Average precision with IOU threshold=.75
    AP-S            Average precision over small objects
                    (< 32px * 32px)
    AR-S            Average recall over small objects
    AP-M            Average precision over medium objects
                    (S < . < 96px * 96px)
    AR-M            Average recall over medium objects
    AP-L            Average precision over large objects
                    (M < . < 1e5px * 1e5px)
    AR-L            Average recall over large objects
                    greater than zero)
    AR-1            Average recall with maximum 1 detection
    AR-10           Average recall with maximum 10 detections
    =============== ==========================================

    .. versionadded:: 0.5.2
    """

    _state_dict_all_req_keys = ("metrics", "ap_50_95")

    ap_50_95: ObjectDetectionAvgPrecisionRecall

    def __init__(
        self,
        num_classes: int = 80,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = True,
    ):
        self.ap_50_95 = ObjectDetectionAvgPrecisionRecall(num_classes=num_classes, device=device)

        super().__init__(
            {
                "S": ObjectDetectionAvgPrecisionRecall(num_classes=num_classes, device=device, area_range="small"),
                "M": ObjectDetectionAvgPrecisionRecall(num_classes=num_classes, device=device, area_range="medium"),
                "L": ObjectDetectionAvgPrecisionRecall(num_classes=num_classes, device=device, area_range="large"),
                "1": ObjectDetectionAvgPrecisionRecall(
                    num_classes=num_classes, device=device, max_detections_per_image_per_class=1
                ),
                "10": ObjectDetectionAvgPrecisionRecall(
                    num_classes=num_classes, device=device, max_detections_per_image_per_class=10
                ),
            },
            output_transform,
            skip_unrolling=skip_unrolling,
        )

    def reset(self) -> None:
        super().reset()
        self.ap_50_95.reset()

    def update(self, output: Sequence[torch.Tensor]) -> None:
        super().update(output)
        self.ap_50_95.update(output)

    def compute(self) -> Dict[str, float]:
        average_precisions_recalls = self.ap_50_95._compute()

        average_precisions_50 = average_precisions_recalls[0, :, 0]
        average_precisions_75 = average_precisions_recalls[0, :, 5]
        if (average_precisions_50 == -1).all():
            AP_50 = AP_75 = AP_50_95 = AR_100 = -1.0
        else:
            AP_50 = average_precisions_50[average_precisions_50 > -1].mean().item()
            AP_75 = average_precisions_75[average_precisions_75 > -1].mean().item()
            AP_50_95 = average_precisions_recalls[0][average_precisions_recalls[0] > -1].mean().item()
            AR_100 = average_precisions_recalls[1][average_precisions_recalls[1] > -1].mean().item()

        result = super().compute()
        return {
            "AP@50..95": AP_50_95,
            "AR-100": AR_100,
            "AP@50": AP_50,
            "AP@75": AP_75,
            "AP-S": result["S"][0],
            "AR-S": result["S"][1],
            "AP-M": result["M"][0],
            "AR-M": result["M"][1],
            "AP-L": result["L"][0],
            "AR-L": result["L"][1],
            "AR-1": result["1"][1],
            "AR-10": result["10"][1],
        }
