__all__ = ["MeanAveragePrecision"]

from typing import Callable, List, Optional, Sequence, Union

import torch
from torchvision.ops import box_iou

from ignite.metrics.metric import Metric


class MeanAveragePrecision(Metric):
    def __init__(
        self,
        iou_thresholds: Optional[Union[List[float], torch.Tensor]] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        if iou_thresholds is None:
            iou_thresholds = torch.range(0.5, 0.99, 0.05)
        self.iou_thresholds = torch.tensor(iou_thresholds, device=device)
        self.rec_thresholds = torch.range(0, 1, 0.01)
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        self._cm = {}

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y, y_pred = output[0].detach(), output[1].detach()
        iou = box_iou(y_pred[:, :4], y[:, :4])
        for iou_thres in self.iou_thresholds:
            iou_thres_item = iou_thres.item()
            valid_iou = torch.clone(iou)
            valid_iou[iou <= iou_thres] = 0
            categories = y[:, 4].unique().tolist()

            for category in categories:
                if category not in self._cm:
                    self._cm[category] = {}
                if iou_thres_item not in self._cm[category]:
                    self._cm[category][iou_thres_item] = {
                        "tp": [],
                        "fp": [],
                        "gt": [],
                        "score": [],
                    }
                class_index_gt = y[:, 4] == category
                class_index_dt = y_pred[:, 5] == category

                class_iou = valid_iou[:, class_index_gt][class_index_dt, :]
                class_iou[~(class_iou == class_iou.max(dim=0)[0])] = 0
                class_iou[~(class_iou.T == class_iou.max(dim=1)[0]).T] = 0

                n_gt = class_index_gt.sum()
                tp = (class_iou != 0).any(dim=1)
                fp = (class_iou == 0).all(dim=1)

                self._cm[category][iou_thres_item]["tp"].append(tp)
                self._cm[category][iou_thres_item]["fp"].append(fp)
                self._cm[category][iou_thres_item]["gt"].append(n_gt)
                self._cm[category][iou_thres_item]["score"].append(y_pred[class_index_dt, 4])

    def compute(self):
        results = []
        for _, cm in self._cm.items():
            category_pr = torch.ones(len(self.iou_thresholds), len(self.rec_thresholds)) * -1
            for idx, (_, cm_iou) in enumerate(cm.items()):
                scores = torch.cat(cm_iou["score"], dim=0)
                indx = torch.argsort(scores, descending=True)
                tp = torch.cat(cm_iou["tp"], dim=0)[indx].cumsum(dim=0)
                fp = torch.cat(cm_iou["fp"], dim=0)[indx].cumsum(dim=0)
                n_gt = sum(cm_iou["gt"])
                rc = tp / n_gt
                pr = tp / (fp + tp)

                for i in range(len(tp) - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                inds = torch.searchsorted(rc, self.rec_thresholds)
                pr_at_recthres = torch.zeros(len(self.rec_thresholds))
                try:
                    for ri, pi in enumerate(inds):
                        pr_at_recthres[ri] = pr[pi]
                except:
                    pass
                category_pr[idx, :] = pr_at_recthres
            category_ap = category_pr[category_pr > -1].mean()
            results.append(category_ap)
        return torch.stack(results).mean()
