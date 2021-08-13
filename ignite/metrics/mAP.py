from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from ignite.metrics.metric import Metric

"""
gt: (1id, 4xmin, 5ymin, 6xmax, 7ymax, 8area, 9crowd, 10ignore)
dt: (1id, 4xmin, 5ymin, 6xmax, 7ymax, 8area, 9crowd, 10confidence)
gts: torch.tensor([gt,gt,gt...])
dts: torch.tensor([dt,dt,dt...])
gt_img : (1img_id, 2class_id, gts)
dt_img : (1img_id, 2class_id, dts)
"""


def iou(gt: torch.Tensor, dt: torch.Tensor, crowd: List) -> torch.Tensor:
    m = len(dt)
    n = len(gt)

    ious = torch.zeros(m, n)

    for g in range(n):
        G = gt[g].tolist()
        ga = G[2] * G[3]
        iscrowd = crowd[g]
        for d in range(m):
            D = dt[d].tolist()
            da = D[2] * D[3]
            ious[m - d - 1, g] = 0
            w = min(D[2] + D[0], G[2] + G[0]) - max(D[0], G[0])
            h = min(D[3] + D[1], G[3] + G[1]) - max(D[1], G[1])
            if w <= 0 or h <= 0:
                continue
            i = w * h
            if iscrowd:
                u = da
            else:
                u = da + ga - i
            ious[m - d - 1, g] = i / u

    return ious


i_id = 0
i_class_id = 1
i_xmin = 2
i_ymin = 3
i_xmax = 4
i_ymax = 5
i_area = 6
i_crowd = 7
g_ignore = 8
d_confidence = 8


class AP(Metric):
    def __init__(
        self,
        area_rngs: Dict,
        max_det: int,
        iou_thrs: torch.Tensor,
        rec_thrs: torch.Tensor,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        self.check_area_rngs(area_rngs)
        self.area_rngs: Dict = area_rngs

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

        super(AP, self).__init__(output_transform=output_transform, device=device)

    @staticmethod
    def check_area_rngs(area_rngs):
        for area in area_rngs:
            area_rng = area_rngs[area]
            if type(area) != str or len(area_rng) != 2 or area_rng[0] >= area_rng[1]:
                raise ValueError(
                    """area_rngs should be a dictionary with key as area name and value as \
                        a tuple of the form (lower limit, upper limit)."""
                )

    @staticmethod
    def get_samples(data):
        if len(data) != 2:
            raise ValueError("Update Data must be of the format [dt_img, gt_img].")

        dt_img, gt_img = data

        if len(dt_img) != 2 or type(dt_img[0]) != int or type(dt_img[1]) != torch.Tensor:
            raise ValueError("Detections should be of the form [image_id, detections_tensor].")
        if len(dt_img[1].shape) != 2 or dt_img[1].shape[1] != 9:
            raise ValueError(f"detections_tensor should be of size [num_detections, 9], got {dt_img[1].shape}")

        if len(gt_img) != 2 or type(gt_img[0]) != int or type(gt_img[1]) != torch.Tensor:
            raise ValueError("Ground Truths should be of the form [image_id, ground_truths].")
        if len(gt_img[1].shape) != 2 or gt_img[1].shape[1] != 9:
            raise ValueError(f"ground_truths should be of size [num_detections, 9], got {gt_img[1].shape}")

        if gt_img[0] != dt_img[0]:
            raise ValueError(
                f"""
            Ground Truth and Detections should be of the same image, got image id {gt_img[0]} \
                for Ground Truth and image id {dt_img[0]} for Detections.
            """
            )

        return dt_img, gt_img

    def update(self, output: Any) -> None:
        dt_img, gt_img = self.get_samples(output)

        dt_img[1] = dt_img[1].to(self.device)
        gt_img[1] = gt_img[1].to(self.device)

        img_id = gt_img[0]

        classes = set()
        classwise_gt: defaultdict = defaultdict(lambda: torch.zeros(0, 9))
        for gt in gt_img[1]:
            class_id = int(gt[i_class_id])
            classes.add(class_id)
            classwise_gt[class_id] = torch.vstack([classwise_gt[class_id], gt])

        classwise_dt: defaultdict = defaultdict(lambda: torch.zeros(0, 9))
        for dt in dt_img[1]:
            class_id = int(dt[i_class_id])
            classes.add(class_id)
            classwise_dt[class_id] = torch.vstack([classwise_dt[class_id], dt])

        self.img_ids.add(img_id)
        self.class_ids.update(classes)

        for class_id in classes:
            ious = self.compute_iou(classwise_gt[class_id], classwise_dt[class_id])
            for area_rng in self.area_rngs:
                self.eval_imgs[class_id, area_rng].append(
                    self.evaluate_image(classwise_gt[class_id], classwise_dt[class_id], self.area_rngs[area_rng], ious)
                )

    def reset(self) -> None:
        self.class_ids: set = set()
        self.img_ids: set = set()

        self.eval_imgs: defaultdict = defaultdict(list)

    def compute_iou(self, gt: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        dt = dt[torch.argsort(dt[:, d_confidence])]

        crowd = [g[i_crowd] for g in gt]
        ious = iou(gt[:, i_xmin:i_area], dt[:, i_xmin:i_area], crowd)

        return ious

    def evaluate_image(
        self, gt: torch.Tensor, dt: torch.Tensor, area_rng: torch.Tensor, ious: torch.Tensor
    ) -> Optional[Dict]:
        if len(gt) == 0 and len(dt) == 0:
            return None

        gt_ignore = torch.zeros(len(gt))
        for i, g in enumerate(gt):
            if g[g_ignore] or (g[i_area] < area_rng[0] or g[i_area] > area_rng[1]):
                gt_ignore[i] = 1
            else:
                gt_ignore[i] = 0

        gt_ind = torch.argsort(gt_ignore)
        gt_ignore = gt_ignore[gt_ind]
        gt = gt[gt_ind]
        dt = dt[torch.argsort(-dt[:, d_confidence])]
        iscrowd = gt[:, i_crowd]

        ious = ious[:, gt_ind] if len(ious) > 0 else ious

        num_iou_thrs = len(self.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)
        gtm = torch.zeros(num_iou_thrs, num_gt)
        dtm = torch.zeros(num_iou_thrs, num_dt)
        dt_ignore = torch.zeros(num_iou_thrs, num_dt)

        if len(ious) != 0:
            for tind, t in enumerate(self.iou_thrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue

                        if m > -1 and gt_ignore[m] == 0 and gt_ignore[gind] == 1:
                            break

                        if ious[dind, gind] < iou:
                            continue

                        iou = ious[dind, gind]
                        m = gind

                    if m == -1:
                        continue
                    dt_ignore[tind, dind] = gt_ignore[m]
                    dtm[tind, dind] = gt[m][i_id]
                    gtm[tind, m] = d[i_id]

        a = torch.tensor([d[i_area] < area_rng[0] or d[i_area] > area_rng[1] for d in dt]).reshape((1, len(dt)))

        dt_ignore = torch.logical_or(
            dt_ignore, torch.logical_and(dtm == 0, torch.repeat_interleave(a, num_iou_thrs, 0))
        )

        return {
            "dtMatches": dtm,
            "dtScores": dt[:, d_confidence],
            "gtIgnore": gt_ignore,
            "dtIgnore": dt_ignore,
        }

    def accumulate(self) -> None:
        num_iou_thr = len(self.iou_thrs)
        num_rec_thr = len(self.rec_thrs)
        num_classes = len(self.class_ids)
        num_area = len(self.area_rngs)

        max_det = self.max_det
        precision = -torch.ones((num_iou_thr, num_rec_thr, num_classes, num_area))

        # retrieve eval_imgs at each category, area range, and max number of detections
        for c, class_id in enumerate(self.class_ids):
            for a, area_rng in enumerate(self.area_rngs):
                # retrieve appropriate eval_imgs from stored results
                eval_imgs = self.eval_imgs[class_id, area_rng]
                eval_imgs = [img for img in eval_imgs if img is not None]
                if len(eval_imgs) == 0:
                    continue
                # Get prediction scores to greedily match
                pred_scores = torch.cat([img["dtScores"][0:max_det] for img in eval_imgs], dim=-1)
                # Sort prediction scores
                inds = torch.argsort(-pred_scores)
                # Retrieve and Sort prediction matches,
                # ignore flags for ground truth and predictions based on prediction scores
                predm = torch.cat([img["dtMatches"][:, 0:max_det] for img in eval_imgs], dim=-1)[:, inds]
                pred_ignore = torch.cat([img["dtIgnore"][:, 0:max_det] for img in eval_imgs], dim=-1)[:, inds]
                gt_ignore = torch.cat([img["gtIgnore"] for img in eval_imgs])
                non_ignored = torch.count_nonzero(gt_ignore == 0)
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

    def summarize_all(self) -> torch.Tensor:
        stats = torch.zeros((6,))
        stats[0] = self._summarize()
        stats[1] = self._summarize(iou_thr=0.5)
        stats[2] = self._summarize(iou_thr=0.75)
        stats[3] = self._summarize(area_rng="small")
        stats[4] = self._summarize(area_rng="medium")
        stats[5] = self._summarize(area_rng="large")

        return stats

    def gather_all(self) -> None:
        import torch.distributed as dist

        def is_dist_avail_and_initialized():
            if not dist.is_available():
                return False
            if not dist.is_initialized():
                return False
            return True

        if not is_dist_avail_and_initialized():
            return

        gather_dicts: List[Dict] = [defaultdict(list)] * dist.get_world_size()
        dist.gather_object(self.eval_imgs, gather_dicts if dist.get_rank() == 0 else None, dst=0)
        if dist.get_rank() == 0:
            keys = set()
            for eval_imgs in gather_dicts:
                for key in eval_imgs:
                    keys.add(key)
            combined_eval_imgs: defaultdict = defaultdict(list)
            for key in keys:
                for eval_imgs in gather_dicts:
                    combined_eval_imgs[key] += eval_imgs[key]
            self.eval_imgs = combined_eval_imgs

    def compute(self) -> torch.Tensor:
        self.gather_all()
        self.accumulate()

        results = self.summarize_all()

        return results
