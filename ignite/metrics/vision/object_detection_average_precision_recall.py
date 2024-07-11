from collections import defaultdict
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union, override

import torch
from typing_extensions import Literal

import ignite.distributed as idist
from ignite.metrics.mean_average_precision import _BaseAveragePrecision, _cat_and_agg_tensors

from ignite.metrics.metric import Metric, reinit__is_reduced
from ignite.metrics.recall import _BasePrecisionRecall


def tensor_list_to_dict_list(
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
    _pred_labels: List[torch.Tensor]
    _P: Dict[int, int]
    _num_classes: int

    def __init__(
        self,
        iou_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        rec_thresholds: Optional[Union[Sequence[float], torch.Tensor]] = None,
        max_detections_per_image: Optional[int] = 100,
        area_range: Optional[Literal["small", "medium", "large", "all"]] = "all",
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        """Calculate the mean average precision & recall for evaluating an object detector.

        In average precision, the maximum precision across thresholds greater or equal a recall threshold is
        considered as the summation operand; In other words, the precision peek across lower or equal
        sensivity levels is used for a recall threshold:

        .. math::
            \text{Average Precision} = \sum_{k=1}^{\#rec\_thresholds} (r_k - r_{k-1}) max(P_{k:})

        Args:
            iou_thresholds: sequence of IoU thresholds to be considered for computing mean average precision & recall.
                Values should be between 0 and 1. If not given, COCO's default (.5, .55, ..., .95) would be used.
            rec_thresholds: sequence of recall thresholds to be considered for computing mean average precision.
                Values should be between 0 and 1. If not given, COCO's default (.0, .01, .02, ..., 1.) would be used.
            max_detections_per_image: Max number of detections in each image to consider for evaluation. The most
                confident ones are selected.
            output_transform: a callable that is used to transform the :class:`~ignite.engine.engine.Engine`'s
                ``process_function``'s output into the form expected by the metric. An already
                provided example is :func:`~ignite.metrics.vision.object_detection_map.tensor_list_to_dict_list`
                which accepts `y_pred` and `y` as lists of tensors and transforms them to the expected format.
                Default is the identity function.
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

        if iou_thresholds is None:
            iou_thresholds = torch.linspace(0.5, 0.95, 10, dtype=torch.double)

        self.iou_thresholds = self._setup_thresholds(iou_thresholds, "iou_thresholds")

        if rec_thresholds is None:
            rec_thresholds = torch.linspace(0, 1, 101, device=device, dtype=torch.double)

        self.area_range = area_range
        self.max_detections_per_image = max_detections_per_image

        super(ObjectDetectionAvgPrecisionRecall, self).__init__(
            output_transform=output_transform,
            device=device,
        )
        super(Metric, self).__init__(
            rec_thresholds=rec_thresholds,
            class_mean=None,
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._tps = []
        self._fps = []
        self._scores = []
        self._pred_labels = []
        self._P = defaultdict(lambda: 0)
        self._num_classes: int = 0

    def _match_area_range(self, bboxes: torch.Tensor) -> torch.Tensor:
        from torchvision.ops.boxes import box_area

        areas = box_area(bboxes)
        if self.area_range == "all":
            min_area = 0
            max_area = 1e10
        elif self.area_range == "small":
            min_area = 0
            max_area = 1024
        elif self.area_range == "medium":
            min_area = 1024
            max_area = 9216
        elif self.area_range == "large":
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
        self, TP: torch.Tensor, FP: torch.Tensor, scores: torch.Tensor, P: torch.Tensor
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

    @override
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
        precision_integrand = precision.flip(-1).cummax(dim=-1).values.flip(-1)
        rec_thresholds = cast(torch.Tensor, self.rec_thresholds).repeat((*recall.shape[:-1], 1))
        rec_thresh_indices = torch.searchsorted(recall, rec_thresholds)
        precision_integrand = precision_integrand.take_along_dim(
            rec_thresh_indices.where(rec_thresh_indices != recall.size(-1), 0), dim=-1
        ).where(rec_thresh_indices != recall.size(-1), 0)
        return torch.sum(precision_integrand, dim=-1) / len(cast(torch.Tensor, self.rec_thresholds))

    def _do_matching(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Tuple[Union[torch.Tensor,None], Union[torch.Tensor,None], Dict[int, int], torch.Tensor, torch.Tensor]:
        r"""
        Matching logic of object detection mAP, according to COCO reference implementation.

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
        labels = target["labels"]
        gt_boxes = target["bbox"]
        gt_is_crowd = (
            target["iscrowd"].bool() if "iscrowd" in target else torch.zeros_like(target["labels"], dtype=torch.bool)
        )
        gt_ignore = ~self._match_area_range(gt_boxes) | gt_is_crowd

        best_detections_index = torch.argsort(pred["scores"], stable=True, descending=True)[
            : self.max_detections_per_image
        ]
        pred_scores = pred["scores"][best_detections_index]
        pred_labels = pred["labels"][best_detections_index]
        pred_boxes = pred["bbox"][best_detections_index]
        pred_match_area_range = self._match_area_range(pred_boxes)

        categories = list(set(labels.int().tolist() + pred_labels.int().tolist()))

        P: Dict[int, int] = {}
        for category in categories:
            category_index_gt = labels == category
            num_category_gt = category_index_gt.sum()
            category_gt_ignore = gt_ignore[category_index_gt]
            if num_category_gt:  # what if P[c] becomes 0 ?
                P[category] = num_category_gt - category_gt_ignore.sum()
        
        if len(pred_labels):
            ious = self.box_iou(pred_boxes, gt_boxes, cast(torch.BoolTensor, gt_is_crowd))
            NO_MATCH = -3
            ious[:, gt_ignore] -= 2
            category_no_match = labels.expand(len(pred_labels), -1) != pred_labels.view(-1, 1)
            ious[category_no_match] = NO_MATCH
            ious.unsqueeze(-1).repeat((1, 1, len(self.iou_thresholds)))
            ious[ious < self.iou_thresholds] = NO_MATCH
            for i in range(len(pred_labels)):
                # Flip is done to give priority to the last item with maximal value, as torch.max selects the first one.
                match_gts = ious[i].flip(0).max(0)
                match_gts_indices = ious.size(1) -1 - match_gts.indices
                for t in range(len(self.iou_thresholds)):
                    if match_gts.values[t] != NO_MATCH and not gt_is_crowd[match_gts_indices[t]]:
                        ious[:, match_gts_indices[t], t] = NO_MATCH
                        ious[i, match_gts_indices[t], t] = match_gts.values[t]

            max_ious = ious.max(1).values
            tp = (max_ious >= 0).T.to(dtype=torch.uint8, device=self._device)
            fp = ((max_ious == NO_MATCH).T and pred_match_area_range).to(dtype=torch.uint8, device=self._device)
        else:
            tp = fp = None            

        return tp, fp, P, pred_scores, pred_labels

    @reinit__is_reduced
    def update(self, output: Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]) -> None:
        r"""Metric update function using prediction and target.

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
                'labels' (N\ :sub:`det`,)   Predicted category number of detections.
                ======== ================== =================================================

                ========= ================= =================================================
                **y items**
                -----------------------------------------------------------------------------
                Key       Value shape        Description
                ========= ================= =================================================
                'bbox'    (N\ :sub:`gt`, 4) Bounding boxes of form (x1, y1, x2, y2)
                                            containing top left and bottom right coordinates.
                'labels'  (N\ :sub:`gt`,)   Category number of ground truths.
                'iscrowd' (N\ :sub:`gt`,)   Whether ground truth boxes are crowd ones or not.
                                            This key is optional.
                ========= ================= =================================================
        """
        self._check_matching_input(output)
        for y_pred, y in zip(*output):
            tp, fp, ps, scores, pred_labels = self._do_matching(y_pred, y)
            if tp is not None:
                self._tps.append(tp.to(device=self._device, dtype=torch.uint8))
                self._fps.append(fp.to(device=self._device, dtype=torch.uint8))
                self._scores.append(scores.to(self._device))
                self._pred_labels.append(pred_labels.to(device=self._device))
            for cls in ps:
                self._P[cls] += ps[cls]
            classes = set(pred_labels.tolist()) | ps.keys()
            if classes:
                self._num_classes = max(max(classes) + 1, self._num_classes)

    def _compute(self) -> Union[torch.Tensor, float]:
        num_classes = int(idist.all_reduce(self._num_classes or 0, "MAX"))
        if num_classes < 1:
            return 0.0

        P = cast(
            torch.Tensor,
            idist.all_reduce(torch.tensor(list(map(self._P.__getitem__, range(num_classes))), device=self._device)),
        )
        if P.sum() < 1:
            return -1.

        pred_labels = _cat_and_agg_tensors(self._pred_labels, (), torch.long, self._device)
        TP = _cat_and_agg_tensors(self._tps, (len(self.iou_thresholds),), torch.uint8, self._device)
        FP = _cat_and_agg_tensors(self._fps, (len(self.iou_thresholds),), torch.uint8, self._device)
        scores = _cat_and_agg_tensors(self._scores, (), torch.double, self._device)

        average_precisions_recalls = -torch.ones(
            (2, num_classes, len(self.iou_thresholds)),
            device=self._device,
            dtype=torch.double,
        )
        for cls in range(num_classes):
            if P[cls] == 0:
                continue
            
            cls_labels = pred_labels == cls
            if sum(cls_labels) == 0:
                average_precisions_recalls[0, cls] = 0.
                continue
            
            recall, precision = self._compute_recall_and_precision(TP[..., cls_labels], FP[..., cls_labels], scores[cls_labels], P[cls])
            average_precision_for_cls_across_other_dims = self._compute_average_precision(recall, precision)
            average_precisions_recalls[0, cls] = average_precision_for_cls_across_other_dims

            average_precisions_recalls[1, cls] = recall[..., -1]
        return average_precisions_recalls
    
    def compute(self) -> Tuple[float, float]:
        average_precisions_recalls = self._compute()
        ap = average_precisions_recalls[0][average_precisions_recalls[0] > -1].mean().item()
        ar = average_precisions_recalls[1][average_precisions_recalls[1] > -1].mean().item()

        return ap, ar


# class ObjDetCommonAPandAR(Metric):
#     """
#     Computes following common variants of average precision (AP) and recall (AR).

#     ============== ======================
#     Metric variant Description
#     ============== ======================
#     AP@.5...95      (..., N\ :sub:`pred`)
#     AP@.5          (N\ :sub:`pred`,)
#     AP@.75              () (A single float,
#     AP-S                greater than zero)
#     AP-M
#     AP-L           (..., \#unique scores)
#     AR-S
#     AR-M
#     AR-L           (..., \#unique scores)
#     ============== ======================
#     """
#     def __init__(self, output_transform: Callable = lambda x:x, device: Union[str, torch.device] = torch.device("cpu")):
#         super().__init__(output_transform, device)