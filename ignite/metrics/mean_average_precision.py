from collections import defaultdict
from typing import Callable, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn import functional as F

import ignite.distributed as idist
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class MeanAveragePrecision(Metric):
    def __init__(
        self,
        iou_thresholds: Optional[Union[List[float], torch.Tensor]] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        r"""Calculate the mean average precision of overall categories.

        Args:
            iou_thresholds: list of IoU thresholds to be considered for computing Mean Average Precision.
                Values should be between 0 and 1. List is sorted internally. Default is that of the COCO
                official evaluation metric.
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
            from torchvision.ops import box_iou

            self.box_iou = box_iou
        except ImportError:
            raise ModuleNotFoundError("This module requires torchvision to be installed.")

        if iou_thresholds is None:
            self.iou_thresholds = [
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.8999999999999999,
                0.95,
            ]  # torch.linspace(0.5, 0.95, 10).tolist()
        elif isinstance(iou_thresholds, torch.Tensor):
            if iou_thresholds.ndim != 1:
                raise ValueError(
                    "`iou_thresholds` should be a one-dimensional tensor or a list of floats"
                    f", given a {iou_thresholds.ndim}-dimensional tensor."
                )
            self.iou_thresholds = iou_thresholds.sort().values.tolist()
        elif isinstance(iou_thresholds, list):
            self.iou_thresholds = iou_thresholds
        else:
            raise TypeError(f"`iou_thresholds` should be a list of floats or a tensor, given {type(iou_thresholds)}.")

        if min(self.iou_thresholds) < 0 or max(self.iou_thresholds) > 1:
            raise ValueError(f"`iou_thresholds` values should be between 0 and 1, given {iou_thresholds}")

        self.rec_thresholds = torch.linspace(0, 1, 101, device=device, dtype=torch.double)
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_categories: int = 0
        self._tp: Dict[int, torch.BoolTensor] = defaultdict(
            lambda: cast(
                torch.BoolTensor, torch.empty((len(self.iou_thresholds), 0), dtype=torch.bool, device=self._device)
            )
        )
        self._num_gt: Dict[int, int] = defaultdict(lambda: 0)
        self._scores: Dict[int, torch.Tensor] = defaultdict(
            lambda: torch.tensor([], dtype=torch.float, device=self._device)
        )

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Args:
            output: a tuple of 2 tensors in which the first one is the prediction and the second is the ground truth.
                The shape of the ground truth is (N, 5) where N stands for the number of ground truth boxes and 5 is
                (x1, y1, x2, y2, class_number). The shape of the prediction is (M, 6) where M stands for the
                number of predicted boxes and 6 is (x1, y1, x2, y2, confidence, class_number).
        """
        y_pred, y = output[0].detach(), output[1].detach()

        if y.ndim != 2 or y.shape[1] != 5:
            raise ValueError(f"Provided y with a wrong shape, expected (N, 5), got {y.shape}")
        if y_pred.ndim != 2 or y_pred.shape[1] != 6:
            raise ValueError(f"Provided y_pred with a wrong shape, expected (M, 6), got {y_pred.shape}")
        # Does coco have class number 0? What is it?
        # Find the number of categories dynamically or by user input?
        categories = torch.cat((y[:, 4], y_pred[:, 5])).unique().int().tolist()
        self._num_categories = max(self._num_categories, max(categories, default=-1) + 1)
        iou = self.box_iou(y_pred[:, :4], y[:, :4])

        for category in categories:
            class_index_gt = y[:, 4] == category
            num_category_gt = class_index_gt.sum()
            self._num_gt[category] += num_category_gt

            class_index_dt = y_pred[:, 5] == category
            if not class_index_dt.any():
                continue

            category_scores = y_pred[class_index_dt, 4]
            self._scores[category] = torch.concat((self._scores[category], category_scores.to(self._device)))

            category_tp = torch.zeros(
                (len(self.iou_thresholds), class_index_dt.sum().item()), dtype=torch.bool, device=self._device
            )
            if class_index_gt.any():
                class_iou = iou[:, class_index_gt][class_index_dt, :]
                category_maximum_iou = class_iou.max()
                category_pred_idx_sorted_by_decreasing_score = torch.argsort(
                    category_scores, stable=True, descending=True
                ).tolist()
                for thres_idx, iou_thres in enumerate(self.iou_thresholds):
                    if iou_thres <= category_maximum_iou:
                        matched_gt_indices = set()
                        for pred_idx in category_pred_idx_sorted_by_decreasing_score:
                            match_iou, match_idx = -1.0, -1
                            for gt_idx in range(num_category_gt):
                                if (class_iou[pred_idx][gt_idx] < iou_thres) or (gt_idx in matched_gt_indices):
                                    continue
                                if class_iou[pred_idx][gt_idx] >= match_iou:
                                    match_iou = class_iou[pred_idx][gt_idx]
                                    match_idx = gt_idx
                            if match_idx != -1:
                                matched_gt_indices.add(match_idx)
                                category_tp[thres_idx][pred_idx] = True
                    else:
                        break

            self._tp[category] = cast(torch.BoolTensor, torch.cat((self._tp[category], category_tp), dim=1))

    @sync_all_reduce("_num_categories:MAX")
    def compute(self) -> float:
        # `gloo` does not support `gather` on GPU. Do we need
        #  to take an action regarding that?
        num_gt = torch.tensor([self._num_gt[cat_id] for cat_id in range(self._num_categories)], device=self._device)
        num_gt = cast(torch.Tensor, idist.all_reduce(num_gt))

        num_predictions = torch.tensor(
            [self._tp[cat_idx].shape[1] for cat_idx in range(self._num_categories)], device=self._device
        )
        world_size = idist.get_world_size()
        if world_size > 1:
            if idist.get_rank() == 0:
                ranks_num_preds = [
                    torch.empty((self._num_categories,), device=self._device, dtype=torch.long)
                    for _ in range(world_size)
                ]
            else:
                ranks_num_preds = None
            dist.gather(num_predictions, ranks_num_preds)
        else:
            ranks_num_preds = [num_predictions]

        max_num_predictions = num_predictions.clone()
        max_num_predictions = cast(torch.Tensor, idist.all_reduce(max_num_predictions, op="MAX"))
        recall_thresh_repeated_iou_thresh_times = self.rec_thresholds.repeat((len(self.iou_thresholds), 1))
        average_precision = torch.tensor(0.0, device=self._device, dtype=torch.double)
        num_present_categories = self._num_categories
        for category_idx in range(self._num_categories):

            if num_gt[category_idx] == 0:
                num_present_categories -= 1
                continue
            if max_num_predictions[category_idx] == 0:
                continue

            if world_size > 1:
                if idist.get_rank() == 0:
                    ranks_tp = [
                        torch.empty(
                            (len(self.iou_thresholds), max_num_predictions[category_idx]),  # type: ignore[arg-type]
                            device=self._device,
                            dtype=torch.uint8,
                        )
                        for _ in range(world_size)
                    ]
                    ranks_scores = [
                        torch.empty((max_num_predictions[category_idx],), device=self._device)  # type: ignore[arg-type]
                        for _ in range(world_size)
                    ]
                else:
                    ranks_tp = None
                    ranks_scores = None
                dist.gather(
                    F.pad(
                        self._tp[category_idx],
                        (
                            0,  # type: ignore[arg-type]
                            max_num_predictions[category_idx] - num_predictions[category_idx],
                        ),
                    ).to(torch.uint8),
                    ranks_tp,
                )
                dist.gather(
                    F.pad(
                        self._scores[category_idx],
                        (
                            0,  # type: ignore[arg-type]
                            max_num_predictions[category_idx] - num_predictions[category_idx],
                        ),
                    ),
                    ranks_scores,
                )
                if idist.get_rank() == 0:
                    ranks_tp = [
                        ranks_tp[r][:, : ranks_num_preds[r][category_idx]].to(torch.bool)  # type: ignore[index, misc]
                        for r in range(world_size)
                    ]
                    tp = torch.cat(ranks_tp, dim=1)

                    ranks_scores = [
                        ranks_scores[r][: ranks_num_preds[r][category_idx]]  # type: ignore[index, misc]
                        for r in range(world_size)
                    ]
                    scores = torch.cat(ranks_scores, dim=0)
            else:
                tp = self._tp[category_idx]
                scores = self._scores[category_idx]
            if idist.get_rank() == 0:

                tp = tp[:, torch.argsort(scores, stable=True, descending=True)]

                tp_summation = tp.cumsum(dim=1).double()
                fp_summation = (~tp).cumsum(dim=1).double()
                recall = tp_summation / num_gt[category_idx]
                precision = tp_summation / (fp_summation + tp_summation + torch.finfo(torch.double).eps)

                recall_thresh_indices = torch.searchsorted(recall, recall_thresh_repeated_iou_thresh_times)
                for t in range(len(self.iou_thresholds)):
                    for r_idx in recall_thresh_indices[t]:
                        if r_idx == recall.shape[1]:
                            break
                        # Interpolated precision. Please refer to PASCAL VOC paper section 4.2
                        # for more information. MS COCO is like PASCAL in this regard.
                        average_precision += precision[t][r_idx:].max()
        if num_present_categories == 0:
            mAP = torch.tensor(-1.0, device=self._device)
        else:
            mAP = average_precision / (num_present_categories * len(self.rec_thresholds) * len(self.iou_thresholds))
        if world_size > 1:
            idist.broadcast(mAP, 0)
        return mAP.item()
