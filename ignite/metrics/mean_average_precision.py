__all__ = ["MeanAveragePrecision"]

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

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
            self.iou_thresholds = torch.arange(0.5, 0.99, 0.05)
        elif isinstance(iou_thresholds, list):
            self.iou_thresholds = torch.tensor(sorted(iou_thresholds), device=device)
        else:
            self.iou_thresholds = iou_thresholds.sort().values
        self.rec_thresholds = torch.linspace(0, 1, 101, device=device)
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_categories: int = 0
        self._tp: Dict[int, torch.BoolTensor] = defaultdict(
            lambda: torch.empty((len(self.iou_thresholds), 0), dtype=torch.bool, device=self._device)
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

        assert y.shape[1] == 5, f"Provided y with a wrong shape, expected (N, 5), got {y.shape}"
        assert y_pred.shape[1] == 6, f"Provided y_pred with a wrong shape, expected (M, 6), got {y.shape}"

        categories = torch.cat((y[:, 4], y_pred[:, 5])).unique().int().tolist()
        self._num_categories = max(self._num_categories, max(categories, default=-1) + 1)
        iou = self.box_iou(y_pred[:, :4], y[:, :4])

        for category in categories:
            class_index_gt = y[:, 4] == category
            self._num_gt[category] += class_index_gt.sum()

            class_index_dt = y_pred[:, 5] == category
            if not class_index_dt.any():
                continue

            self._scores[category] = torch.concat((self._scores[category], y_pred[class_index_dt, 4]))

            category_tp = torch.zeros(
                (len(self.iou_thresholds), class_index_dt.sum()), dtype=torch.bool, device=self._device
            )
            if class_index_gt.any():
                class_iou = iou[:, class_index_gt][class_index_dt, :]
                category_maximum_iou = class_iou.max()
                for thres_idx, iou_thres in enumerate(self.iou_thresholds):
                    if iou_thres < category_maximum_iou:
                        class_iou[class_iou <= iou_thres] = 0
                        class_iou[~(class_iou == class_iou.amax(dim=0))] = 0
                        class_iou[~(class_iou.T == class_iou.amax(dim=1)).T] = 0

                        category_tp[thres_idx] = (class_iou != 0).any(dim=1)
                    else:
                        break

            self._tp[category] = torch.cat((self._tp[category], category_tp), dim=1)

    @sync_all_reduce("_num_categories:MAX")
    def compute(self) -> float:
        # `gloo` does not support `gather` on GPU. Do we need
        #  to take an action regarding that?
        num_gt = torch.tensor([self._num_gt[cat_id] for cat_id in range(self._num_categories)], device=self._device)
        dist.reduce(num_gt)

        num_predictions = torch.tensor(
            [self._tp[cat_idx].shape[1] for cat_idx in range(self._num_categories)], device=self._device
        )
        if idist.get_local_rank() == 0:
            ranks_num_predictions = [
                torch.empty((self._num_categories,), device=self._device) for _ in range(idist.get_world_size())
            ]
        else:
            ranks_num_predictions = None
        dist.gather(num_predictions, ranks_num_predictions)
        max_num_predictions = idist.all_reduce(num_predictions, op="MAX")
        recall_thresh_repeated_iou_thresh_times = self.rec_thresholds.repeat((len(self.iou_thresholds), 1))
        AP = []
        for category_idx in range(self._num_categories):

            if num_gt[category_idx] == 0:
                continue

            if idist.get_local_rank() == 0:
                ranks_tp = [
                    torch.empty((len(self.iou_thresholds), max_num_predictions[category_idx]), device=self._device)
                    for _ in range(idist.get_world_size())
                ]
                ranks_scores = [
                    torch.empty((max_num_predictions[category_idx],), device=self._device)
                    for _ in range(idist.get_world_size())
                ]
            else:
                ranks_tp = None
                ranks_scores = None
            dist.gather(
                F.pad(self._tp[category_idx], (0, max_num_predictions[category_idx] - num_predictions[category_idx])),
                ranks_tp,
            )
            dist.gather(
                F.pad(
                    self._scores[category_idx], (0, max_num_predictions[category_idx] - num_predictions[category_idx])
                ),
                ranks_scores,
            )
            if idist.get_local_rank() == 0:
                ranks_tp = [
                    ranks_tp[r][:, : ranks_num_predictions[r][category_idx]] for r in range(idist.get_world_size())
                ]
                tp = torch.cat(ranks_tp, dim=1)

                ranks_scores = [
                    ranks_scores[r][:, : ranks_num_predictions[r][category_idx]] for r in range(idist.get_world_size())
                ]
                scores = torch.cat(ranks_scores, dim=1)

                tp = tp[:, torch.argsort(scores, descending=True)]

                tp = tp.cumsum(dim=1)
                fp = (~tp).cumsum(dim=1)
                recall = tp / num_gt[category_idx]
                precision = tp / (fp + tp)

                interpolated_precision_at_recall_thresh = []
                recall_thresh_indices = torch.searchsorted(recall, recall_thresh_repeated_iou_thresh_times)
                for t in range(len(self.iou_thresholds)):
                    for r_idx in recall_thresh_indices:
                        if r_idx == recall.shape[1]:
                            break
                        interpolated_precision_at_recall_thresh.append(precision[t][r_idx:].max())
                AP.append(
                    sum(interpolated_precision_at_recall_thresh) / (len(self.rec_thresholds) * len(self.iou_thresholds))
                )
        if idist.get_local_rank() == 0:
            mAP = torch.tensor(AP, device=self._device).mean()
        else:
            mAP = torch.tensor(0.0, device=self._device)
        dist.broadcast(mAP, 0)
        return mAP.item()
