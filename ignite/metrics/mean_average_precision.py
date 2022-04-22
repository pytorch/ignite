__all__ = ["MeanAveragePrecision"]

from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import torch

from ignite.metrics.metric import Metric

box_iou = None


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
        global box_iou
        try:
            from torchvision.ops import box_iou
        except ImportError:
            raise RuntimeError("This module requires torchvision to be installed.")

        if iou_thresholds is None:
            iou_thresholds = torch.range(0.5, 0.99, 0.05)
        self.iou_thresholds = torch.tensor(iou_thresholds, device=device)
        self.rec_thresholds = torch.linspace(0, 1, 101, device=device)
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        self._cm = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """
        Args:
            output: a tuple of 2 tensors in which the first one is the truth and the second one is the prediction.
                the shape of the output[0](truth) is (N, 5) where N stands for the number of ground truth boxes and 5 is
                (x1, x2, y1, y2, class_number); the shape of the output[1](prediction) is (M, 6) where M stands for the
                number of predicted boxes and 6 is (x1, x2, y1, y2, confidence, class_number).
        """
        y, y_pred = output[0].detach(), output[1].detach()

        assert y.shape[1] == 5, f"Provided y with a wrong shape, expected (N, 5), got {y.shape}"
        assert y_pred.shape[1] == 6, f"Provided y_hat with a wrong shape, expected (N, 6), got {y.shape}"

        iou = box_iou(y_pred[:, :4], y[:, :4])
        for iou_thres in self.iou_thresholds:
            iou_thres_item = iou_thres.item()
            valid_iou = torch.clone(iou)
            valid_iou[iou <= iou_thres] = 0
            categories = torch.cat((y[:, 4], y_pred[:, 5])).unique().tolist()

            for category in categories:
                class_index_gt = y[:, 4] == category
                class_index_dt = y_pred[:, 5] == category

                class_iou = valid_iou[:, class_index_gt][class_index_dt, :]

                if class_iou.shape[1] == 0:
                    # no ground truth of the category
                    n_gt = 0
                    tp = torch.tensor([False] * class_iou.shape[0], device=self._device)
                    fp = torch.tensor([True] * class_iou.shape[0], device=self._device)
                    score = y_pred[class_index_dt, 4]
                elif class_iou.shape[0] == 0:
                    # no predictions of the category
                    n_gt = class_iou.shape[1]
                    tp = torch.tensor([], device=self._device).bool()
                    fp = torch.tensor([], device=self._device).bool()
                    score = torch.tensor([], device=self._device)
                else:
                    class_iou[~(class_iou == class_iou.max(dim=0)[0])] = 0
                    class_iou[~(class_iou.T == class_iou.max(dim=1)[0]).T] = 0

                    n_gt = class_iou.shape[1]
                    tp = (class_iou != 0).any(dim=1)
                    fp = (class_iou == 0).all(dim=1)
                    score = y_pred[class_index_dt, 4]

                self._cm[category][iou_thres_item]["tp"].append(tp)
                self._cm[category][iou_thres_item]["fp"].append(fp)
                self._cm[category][iou_thres_item]["gt"].append(n_gt)
                self._cm[category][iou_thres_item]["score"].append(score)

    def compute(self):
        results = []
        for _, cm in self._cm.items():
            category_pr = torch.ones(len(self.iou_thresholds), len(self.rec_thresholds), device=self._device) * -1

            for idx, (_, cm_iou) in enumerate(cm.items()):
                n_gt = sum(cm_iou["gt"])
                if n_gt == 0:
                    # no ground truth of the class
                    continue
                scores = torch.cat(cm_iou["score"], dim=0)
                indx = torch.argsort(scores, descending=True)
                tp = torch.cat(cm_iou["tp"], dim=0)[indx].cumsum(dim=0)
                fp = torch.cat(cm_iou["fp"], dim=0)[indx].cumsum(dim=0)
                rc = tp / n_gt
                pr = tp / (fp + tp)

                for i in range(len(tp) - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                inds = torch.searchsorted(rc, self.rec_thresholds)
                pr_at_recthres = torch.zeros(len(self.rec_thresholds), device=self._device)
                try:
                    for ri, pi in enumerate(inds):
                        pr_at_recthres[ri] = pr[pi]
                except:
                    pass
                category_pr[idx, :] = pr_at_recthres
            if torch.all(category_pr == -1):
                continue
            category_ap = category_pr[category_pr > -1].mean()
            results.append(category_ap)
        return torch.stack(results).mean().item()
