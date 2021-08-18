from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch

from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


def iou(y: torch.Tensor, y_pred: torch.Tensor, crowd: List) -> torch.Tensor:
    m = len(y_pred)
    n = len(y)

    ious = torch.zeros(m, n)

    for g in range(n):
        Y = y[g].tolist()
        ya = Y[2] * Y[3]
        iscrowd = crowd[g]
        for d in range(m):
            P = y_pred[d].tolist()
            pa = P[2] * P[3]
            ious[m - d - 1, g] = 0
            w = min(P[2] + P[0], Y[2] + Y[0]) - max(P[0], Y[0])
            h = min(P[3] + P[1], Y[3] + Y[1]) - max(P[1], Y[1])
            if w <= 0 or h <= 0:
                continue
            i = w * h
            if iscrowd:
                u = pa
            else:
                u = pa + ya - i
            ious[m - d - 1, g] = i / u

    return ious


class MeanAveragePrecision(Metric):
    r"""Calculates Mean Average Precision for object detection data.

    .. math::
       \text{MeanAveragePrecision} = {1/11}\sum_{r \in {0.0, ..., 1.0}}\text{AP}_r

    where \text{MeanAveragePrecision}_r represents average precision at recall value :math:`r`,
    where :math:`r` ranges from 0.0 to 1.0.

    More details can be found in `Everingham et al. 2009`__

    __ https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

    Remark:

        This implementation is inspired by pycocotools which can be found `here`__

        __ https://github.com/cocodataset/cocoapi/

    Args:
        area_rngs: dictionary with area_name as key and a tuple of the form
            (lower_area_range, upper_area_range) as value, where both values are floats.
            It contains all area ranges for which AP needs to be computed.
        max_det: Maximum detections allowed per image. This is to limit crowding and repeated
            detections of the same object.
        iou_thrs: A float tensor containing a set of IoU thresholds for which AP needs to be computed.
        rec_thrs: A float tensor containing a set of values at which AP needs to be computed for averaging.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Example:

        .. code-block:: python

            import torch
            from ignite.metrics import MeanAveragePrecision

            # Detection Format:
            # (id, class, xmin, ymin, wiy_predh, height, area, crowd, ignore/confidence)

            # ground truth detections for an image
            ys = torch.tensor([[1,73,249.1,50.7,375.3,277.2,101928.52,0,0],
                    [2,73,152.3,50.29,47,312.38, 4097.4,0,0]])

            # predicted detections for the same image
            y_preds = torch.tensor([[1,73,176.38,50.3,463.62,312.4,144825.6,0,0.328],
                    [2,73,355.5,111.87,210,93.75,19687.5,0,0.64]])

            mAP = MeanAveragePrecision()

            mAP.update([ys, y_preds])

            mAP.compute()

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        area_rngs: Dict = {
            "all": [0, float("inf")],
            "small": [0, 1024],
            "medium": [1024, 9216],
            "large": [9216, float("inf")],
        },
        max_det: int = 100,
        iou_thrs: torch.Tensor = torch.tensor([0.5 + (i / 20) for i in range(10)]),  # type: ignore
        rec_thrs: torch.Tensor = torch.tensor([i / 100 for i in range(101)]),  # type: ignore
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        self._check_area_rngs(area_rngs)
        self.area_rngs: Dict = area_rngs
        self.area_rngs["all"] = [0, float("inf")]

        self.max_det: int = max_det
        if type(self.max_det) != int or self.max_det < 1:
            raise ValueError(f"max_det should be a positive integer, got {self.max_det}")

        if not torch.is_floating_point(iou_thrs):
            raise ValueError(f"iou_thrs should be a float tensor, got {iou_thrs.dtype}")
        self.iou_thrs: torch.Tensor = iou_thrs.to(device)

        if not torch.is_floating_point(rec_thrs):
            raise ValueError(f"rec_thrs should be a float tensor, got {rec_thrs.dtype}")
        self.rec_thrs: torch.Tensor = rec_thrs.to(device)

        self.device: Union[str, torch.device] = device

        class labels:
            i_id = 0
            i_category_id = 1
            i_xmin = 2
            i_area = 6
            i_crowd = 7
            g_ignore = 8
            d_confidence = 8

        self.labels = labels

        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    @staticmethod
    def _check_area_rngs(area_rngs: Dict) -> None:
        for area in area_rngs:
            area_rng = area_rngs[area]
            if type(area) != str or len(area_rng) != 2 or area_rng[0] >= area_rng[1]:
                raise ValueError(
                    """area_rngs should be a dictionary with key as area name and value as \
                        a tuple of the form (lower limit, upper limit)."""
                )

    @staticmethod
    def _get_samples(data: List) -> Tuple:
        if len(data) != 2:
            raise ValueError("Update Data must be of the format [y_pred_img, y_img].")

        y_pred_img, y_img = data

        if len(y_pred_img.shape) != 2 or y_pred_img.shape[1] != 9:
            raise ValueError(f"detections_tensor should be of size [num_detections, 9], got {y_pred_img.shape}")

        if len(y_img.shape) != 2 or y_img.shape[1] != 9:
            raise ValueError(f"ground_truths should be of size [num_detections, 9], got {y_img.shape}")

        return y_pred_img, y_img

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        y_pred_img, y_img = self._get_samples(output)

        y_pred_img = y_pred_img.to(self.device)
        y_img = y_img.to(self.device)

        categories = set()
        categorywise_y: defaultdict = defaultdict(lambda: torch.zeros(0, 9).to(self.device))
        for y in y_img:
            category_id = int(y[self.labels.i_category_id])
            categories.add(category_id)
            categorywise_y[category_id] = torch.vstack([categorywise_y[category_id], y])

        categorywise_y_pred: defaultdict = defaultdict(lambda: torch.zeros(0, 9).to(self.device))
        for y_pred in y_pred_img:
            category_id = int(y_pred[self.labels.i_category_id])
            categories.add(category_id)
            categorywise_y_pred[category_id] = torch.vstack([categorywise_y_pred[category_id], y_pred])

        self.category_ids.update(categories)

        for category_id in categories:
            ious = self._compute_iou(categorywise_y[category_id], categorywise_y_pred[category_id])
            for area_rng in self.area_rngs:
                self.eval_imgs[category_id, area_rng].append(
                    self._evaluate_image(
                        categorywise_y[category_id], categorywise_y_pred[category_id], self.area_rngs[area_rng], ious
                    )
                )

    @reinit__is_reduced
    def reset(self) -> None:
        self.category_ids: set = set()

        self.eval_imgs: defaultdict = defaultdict(list)

    def _compute_iou(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred[torch.argsort(y_pred[:, self.labels.d_confidence])]

        crowd = [g[self.labels.i_crowd] for g in y]
        ious = iou(
            y[:, self.labels.i_xmin : self.labels.i_area], y_pred[:, self.labels.i_xmin : self.labels.i_area], crowd
        ).to(self.device)

        return ious

    def _evaluate_image(
        self, y: torch.Tensor, y_pred: torch.Tensor, area_rng: torch.Tensor, ious: torch.Tensor
    ) -> Optional[Dict]:
        if len(y) == 0 and len(y_pred) == 0:
            return None

        y_ignore = torch.zeros(len(y))
        for i, g in enumerate(y):
            if g[self.labels.g_ignore] or (g[self.labels.i_area] < area_rng[0] or g[self.labels.i_area] > area_rng[1]):
                y_ignore[i] = 1
            else:
                y_ignore[i] = 0

        y_ind = torch.argsort(y_ignore)
        y_ignore = y_ignore[y_ind]
        y = y[y_ind]
        y_pred = y_pred[torch.argsort(-y_pred[:, self.labels.d_confidence])]
        iscrowd = y[:, self.labels.i_crowd]

        ious = ious[:, y_ind] if len(ious) > 0 else ious

        num_iou_thrs = len(self.iou_thrs)
        num_y = len(y)
        num_y_pred = len(y_pred)
        ym = torch.zeros((num_iou_thrs, num_y), device=self.device)
        y_predm = torch.zeros((num_iou_thrs, num_y_pred), device=self.device)
        y_pred_ignore = torch.zeros((num_iou_thrs, num_y_pred), device=self.device)

        if len(ious) != 0:
            for tind, t in enumerate(self.iou_thrs):
                for dind, d in enumerate(y_pred):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(y):
                        if ym[tind, gind] > 0 and not iscrowd[gind]:
                            continue

                        if m > -1 and y_ignore[m] == 0 and y_ignore[gind] == 1:
                            break

                        if ious[dind, gind] < iou:
                            continue

                        iou = ious[dind, gind]
                        m = gind

                    if m == -1:
                        continue
                    y_pred_ignore[tind, dind] = y_ignore[m]
                    y_predm[tind, dind] = y[m][self.labels.i_id]
                    ym[tind, m] = d[self.labels.i_id]

        a = torch.tensor(
            [d[self.labels.i_area] < area_rng[0] or d[self.labels.i_area] > area_rng[1] for d in y_pred]
        ).reshape((1, len(y_pred)))
        a = a.to(self.device)

        y_pred_ignore = torch.logical_or(
            y_pred_ignore, torch.logical_and(y_predm == 0, torch.repeat_interleave(a, num_iou_thrs, 0))
        ).to(self.device)

        return {
            "y_pred_matches": y_predm,
            "y_pred_scores": y_pred[:, self.labels.d_confidence],
            "y_ignore": y_ignore,
            "y_pred_ignore": y_pred_ignore,
        }

    @reinit__is_reduced
    def _accumulate(self) -> None:
        num_iou_thr = len(self.iou_thrs)
        num_rec_thr = len(self.rec_thrs)
        num_categories = len(self.category_ids)
        num_area = len(self.area_rngs)

        max_det = self.max_det
        precision = -torch.ones((num_iou_thr, num_rec_thr, num_categories, num_area), device=self.device)

        # retrieve eval_imgs at each category, area range, and max number of detections
        for c, category_id in enumerate(self.category_ids):
            for a, area_rng in enumerate(self.area_rngs):
                # retrieve appropriate eval_imgs from stored results
                eval_imgs = self.eval_imgs[category_id, area_rng]
                eval_imgs = [img for img in eval_imgs if img is not None]
                if len(eval_imgs) == 0:
                    continue
                # Get prediction scores to greedily match
                pred_scores = torch.cat([img["y_pred_scores"][0:max_det] for img in eval_imgs], dim=-1)
                # Sort prediction scores
                inds = torch.argsort(-pred_scores)
                # Retrieve and Sort prediction matches,
                # ignore flags for ground truth and predictions based on prediction scores
                predm = torch.cat([img["y_pred_matches"][:, 0:max_det] for img in eval_imgs], dim=-1)[:, inds]
                pred_ignore = torch.cat([img["y_pred_ignore"][:, 0:max_det] for img in eval_imgs], dim=-1)[:, inds]
                y_ignore = torch.cat([img["y_ignore"] for img in eval_imgs])
                non_ignored = torch.count_nonzero(y_ignore == 0)
                if non_ignored == 0:
                    continue
                # Calculate true positive and false positive based on prediction matches and ignore flags
                tps = torch.logical_and(predm, torch.logical_not(pred_ignore))
                fps = torch.logical_and(torch.logical_not(predm), torch.logical_not(pred_ignore))
                # Calculate precision and recall iteratively
                tp_sum = torch.cumsum(tps, dim=1).to(device=self.device, dtype=torch.float64)
                fp_sum = torch.cumsum(fps, dim=1).to(device=self.device, dtype=torch.float64)

                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    nd = len(tp)
                    rc = tp / non_ignored
                    pr = tp / (fp + tp + torch.finfo(torch.float64).eps)
                    q: Any = torch.zeros((num_rec_thr,))

                    pr = pr.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    inds = torch.searchsorted(rc, self.rec_thrs, right=False)
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                    except:
                        pass
                    precision[t, :, c, a] = torch.tensor(q)
        # return precision, recall
        self.precision = precision

    def _summarize(self, iou_thr: Optional[float] = None, area_rng: str = "all") -> float:
        aind = [i for i, a_rng in enumerate(self.area_rngs) if a_rng == area_rng]
        # Calculate Average Precision
        s = self.precision
        if iou_thr is not None:
            t = (self.iou_thrs == iou_thr).int().nonzero(as_tuple=True)[0]
            s = s[t]
        s = s[:, :, :, aind]
        # Take mean to calculate mAP
        if len(s[s > -1]) == 0:
            mean_s = torch.tensor(-1)
        else:
            mean_s = torch.mean(s[s > -1])

        return mean_s.item()

    @reinit__is_reduced
    def _gather_all(self) -> None:
        import torch.distributed as dist

        def is_dist_avail_and_initialized() -> bool:
            if not dist.is_available():
                return False
            if not dist.is_initialized():
                return False
            return True

        if not is_dist_avail_and_initialized():
            return

        eval_gather_dicts: List[Dict] = [defaultdict(list)] * dist.get_world_size()
        dist.all_gather_object(eval_gather_dicts, self.eval_imgs)

        category_gather_list: List[Set] = [set()] * dist.get_world_size()
        dist.all_gather_object(category_gather_list, self.category_ids)

        keys = set()
        for eval_imgs in eval_gather_dicts:
            for key in eval_imgs:
                keys.add(key)
        combined_eval_imgs: defaultdict = defaultdict(list)
        for key in keys:
            for eval_imgs in eval_gather_dicts:
                combined_eval_imgs[key] += eval_imgs[key]
        self.eval_imgs = combined_eval_imgs

        all_category_ids = set()
        for category_set in category_gather_list:
            for category_id in category_set:
                all_category_ids.add(category_id)

        self.category_ids = all_category_ids

    @sync_all_reduce()
    def compute(self) -> Dict:
        self._gather_all()
        self._accumulate()

        results = {
            "all@0.5": self._summarize(iou_thr=0.5),
            "all@0.75": self._summarize(iou_thr=0.75),
        }

        for area in self.area_rngs:
            results[area] = self._summarize(area_rng=area)

        return results
