from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch

from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MeanAveragePrecision"]


def _iou(y: torch.Tensor, y_pred: torch.Tensor, crowd: List) -> torch.Tensor:
    m = len(y_pred)
    n = len(y)

    ious = torch.zeros(m, n)

    # bbox format : (xmin, ymin, width, height)
    for g in range(n):
        y_bbox = y[g].tolist()
        y_area = y_bbox[2] * y_bbox[3]
        if crowd is not None:
            iscrowd = crowd[g]
        else:
            iscrowd = 0
        for d in range(m):
            y_pred_bbox = y_pred[d].tolist()
            y_pred_area = y_pred_bbox[2] * y_pred_bbox[3]
            ious[d, g] = 0
            w = min(y_pred_bbox[2] + y_pred_bbox[0], y_bbox[2] + y_bbox[0]) - max(y_pred_bbox[0], y_bbox[0])
            h = min(y_pred_bbox[3] + y_pred_bbox[1], y_bbox[3] + y_bbox[1]) - max(y_pred_bbox[1], y_bbox[1])
            if w <= 0 or h <= 0:
                continue
            intersection = w * h
            if iscrowd:
                union = y_pred_area
            else:
                union = y_pred_area + y_area - intersection
            ious[d, g] = intersection / union

    return ious


class MeanAveragePrecision(Metric):
    r"""Calculates Mean Average Precision (mAP) for object detection data.

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
        object_area_ranges: dictionary with area_name as key and a tuple of the form
            (lower_area_range, upper_area_range) as value, where both values are floats.
            It contains all area ranges for which AP needs to be computed.
        num_detection_max: Number of maximum detections allowed per image. This is to limit crowding and repeated
            detections of the same object.
        iou_thresholds: Optional list of float IoU thresholds for which AP is computed (default: [.5:.05:.95]).
        rec_thresholds: Optional list of float values to which AP is computed for averaging (default: [0:.01:1]).
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

            # Input Format:
            # Ground Truth
            # [
            #    {
            #        "image_id": torch.IntTensor(B),
            #        "category_id": torch.IntTensor(B),
            #        "bbox": torch.FloatTensor(B x 4),
            #        "iscrowd": torch.IntTensor(B) (Optional),
            #        "area": torch.FloatTensor(B) (Optional),
            #        "ignore": torch.IntTensor(B) (Optional),
            #    }
            # ]

            # Prediction
            # [
            #    {
            #        "image_id": torch.IntTensor(B),
            #        "category_id": torch.IntTensor(B),
            #        "bbox": torch.FloatTensor(B x 4),
            #        "score": torch.FloatTensor(B),
            #    }
            # ]

            ys = {'area': tensor([2.]),
                          'bbox': tensor([[0., 0., 2., 1.]]),
                          'category_id': tensor([2]),
                          'id': tensor([2]),
                          'ignore': tensor([0]),
                          'image_id': 1,
                          'iscrowd': tensor([0])}

            y_preds = {'bbox': tensor([[0., 0., 2., 1.]]),
                          'category_id': tensor([2]),
                          'id': tensor([2]),
                          'image_id': 1,
                          'score': tensor([0.8999999762])}

            mAP = MeanAveragePrecision()

            mAP.update([ys, y_preds])

            mAP.compute()

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        object_area_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        num_detection_max: int = 100,
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        if object_area_ranges is None:
            self.object_area_ranges = {
                "all": (0.0, float("inf")),
                "small": (0.0, 1024.0),
                "medium": (1024.0, 9216.0),
                "large": (9216.0, float("inf")),
            }
        else:
            self.object_area_ranges = object_area_ranges
            if "all" not in self.object_area_ranges:
                self.object_area_ranges["all"] = (0, float("inf"))

        self._check_object_area_ranges()

        if num_detection_max < 1:
            raise ValueError(f"Argument num_detection_max should be a positive integer, got {num_detection_max}")

        self.num_detection_max = num_detection_max

        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.iou_thresholds = torch.tensor(iou_thresholds, device=device)

        if rec_thresholds is None:
            rec_thresholds = [i / 100 for i in range(101)]
        self.rec_thresholds = torch.tensor(rec_thresholds, device=device)

        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    def _check_object_area_ranges(self) -> None:
        for area, area_range in self.object_area_ranges.items():
            if len(area_range) != 2 or area_range[0] >= area_range[1]:
                raise ValueError(
                    f"object_area_ranges must be a dict associating to each key (str) a tuple of float values (a, b) "
                    f"where a < b (got: key={area}, value={area_range})"
                )

    @reinit__is_reduced
    def update(self, outputs: Tuple[Dict, Dict]) -> None:

        for output in outputs:

            y_pred_img, y_img = output

            assert y_img["image_id"] == y_pred_img["image_id"]

            if y_img["image_id"] in self.image_ids:
                raise ValueError("Detections for this image_id are already evaluated.")

            self.image_ids.add(y_img["image_id"])

            y_category_dict = defaultdict(list)
            for i, category_id in enumerate(y_img["category_id"]):
                y_category_dict[int(category_id)].append(i)

            y_pred_category_dict = defaultdict(list)
            for i, category_id in enumerate(y_pred_img["category_id"]):
                y_pred_category_dict[int(category_id)].append(i)

            categories = torch.unique(torch.cat([y_img["category_id"], y_pred_img["category_id"]])).tolist()
            self.category_ids.update(categories)

            for category in categories:
                y_pred_ind = y_pred_category_dict[category]
                y_pred_bbox = y_pred_img["bbox"][y_pred_ind]
                y_pred_score = y_pred_img["score"][y_pred_ind]
                y_pred_id = y_pred_img["id"][y_pred_ind]
                y_pred_area = (y_pred_img["bbox"][:, 2] * y_pred_img["bbox"][:, 3])[y_pred_ind]

                sorted_y_pred_bbox = y_pred_bbox[torch.argsort(-y_pred_img["score"][y_pred_ind])]

                y_ind = y_category_dict[category]
                y_id = y_img["id"][y_ind]
                y_bbox = y_img["bbox"][y_ind]
                crowd = y_img["iscrowd"][y_ind] if "iscrowd" in y_img else None
                y_ignore = y_img["ignore"][y_ind] if "ignore" in y_img else torch.zeros(len(y_ind))
                y_area = y_img["area"][y_ind] if "area" in y_img else (y_img["bbox"][:, 2] * y_img["bbox"][:, 3])[y_ind]

                ious = _iou(y_bbox, sorted_y_pred_bbox, crowd).to(self._device)
                for area_rng in self.object_area_ranges:
                    eval_img = self._evaluate_image_matches(
                        [y_id, y_area, y_ignore, crowd],
                        [y_pred_id, y_pred_area, y_pred_score],
                        self.object_area_ranges[area_rng],
                        ious,
                    )

                    if eval_img is not None:
                        self.eval_imgs[int(category), area_rng].append(eval_img)

    @reinit__is_reduced
    def reset(self) -> None:
        self.category_ids: set = set()
        self.eval_imgs: defaultdict = defaultdict(list)
        self.image_ids: set = set()

    def _evaluate_image_matches(
        self, y: List, y_pred: List, area_rng: Tuple[float, float], ious: torch.Tensor
    ) -> Optional[Dict]:
        """
            Evaluates iou_threshold wise y and y_pred matches.
        """

        y_id, y_area, y_ignore, y_crowd = y
        y_pred_id, y_pred_area, y_pred_score = y_pred

        if len(y_id) == 0 and len(y_pred_id) == 0:
            return None

        # assign which detections in y are ignored for evaluating matches
        ignore = torch.zeros(len(y_id))
        for i, area in enumerate(y_area):
            if y_ignore[i] == 1 or area < area_rng[0] or area > area_rng[1]:
                ignore[i] = 1

        # Sort y based such that non ignored predictions are at the start
        y_ind = torch.argsort(ignore)
        y_id = y_id[y_ind]
        ignore = ignore[y_ind]
        y_area = y_area[y_ind]

        # Sort y_pred according to confidence since we are using a greedy matching approach
        y_pred_ind = torch.argsort(-y_pred_score)
        y_pred_id = y_pred_id[y_pred_ind]
        y_pred_area = y_pred_area[y_pred_ind]
        y_pred_score = y_pred_score[y_pred_ind]

        # Sort ious accordingly
        ious = ious[:, y_ind] if len(ious) > 0 else ious

        num_iou_thrs = len(self.iou_thresholds)
        num_y = len(y_ind)
        num_y_pred = len(y_pred_ind)
        ym = torch.zeros((num_iou_thrs, num_y), device=self._device)
        y_predm = torch.zeros((num_iou_thrs, num_y_pred), device=self._device)
        y_pred_ignore = torch.zeros((num_iou_thrs, num_y_pred), device=self._device)

        if len(ious) != 0:
            for tind, t in enumerate(self.iou_thresholds):
                for dind, d in enumerate(y_pred_id):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(y_id):
                        # Find the best ground truth match for a prediction based on ious.
                        if ym[tind, gind] > 0 and not y_crowd[gind]:
                            continue

                        if m > -1 and ignore[m] == 0 and ignore[gind] == 1:
                            break

                        if ious[dind, gind] < iou:
                            continue

                        iou = ious[dind, gind]
                        m = gind

                    if m == -1:
                        continue
                    y_pred_ignore[tind, dind] = ignore[m]
                    y_predm[tind, dind] = y_id[m]
                    ym[tind, m] = d

        # Sort the results area_wise, helps in future calculation of areawise mAP.
        d_area_ignore = torch.zeros(len(y_pred_ind))
        for i, area in enumerate(y_pred_area):
            if area < area_rng[0] or area > area_rng[1]:
                d_area_ignore[i] = 1
            else:
                d_area_ignore[i] = 0

        a = d_area_ignore.reshape((1, len(y_pred_ind)))
        a = a.to(self._device)

        y_pred_ignore = torch.logical_or(
            y_pred_ignore, torch.logical_and(y_predm == 0, torch.repeat_interleave(a, num_iou_thrs, 0))
        ).to(self._device)

        return {"matches": y_predm, "scores": y_pred_score, "ignore": {"y": ignore, "y_pred": y_pred_ignore,}}

    def _accumulate(self) -> None:
        num_iou_thr = len(self.iou_thresholds)
        num_rec_thr = len(self.rec_thresholds)
        num_categories = len(self.category_ids)
        num_area = len(self.object_area_ranges)

        precision = -torch.ones((num_iou_thr, num_rec_thr, num_categories, num_area), device=self._device)

        # retrieve eval_imgs at each category, area range, and max number of detections
        for c, category_id in enumerate(self.category_ids):
            for a, area_rng in enumerate(self.object_area_ranges):
                # retrieve appropriate eval_imgs from stored results
                eval_imgs = self.eval_imgs[category_id, area_rng]
                eval_imgs = [img for img in eval_imgs if img is not None]
                if len(eval_imgs) == 0:
                    continue
                # Get prediction scores to greedily match
                pred_scores = torch.cat([img["scores"][0 : self.num_detection_max] for img in eval_imgs], dim=-1)
                # Sort prediction scores
                inds = torch.argsort(-pred_scores)
                # Retrieve and Sort prediction matches,
                # ignore flags for ground truth and predictions based on prediction scores
                predm = torch.cat([img["matches"][:, 0 : self.num_detection_max] for img in eval_imgs], dim=-1)[:, inds]
                pred_ignore = torch.cat(
                    [img["ignore"]["y_pred"][:, 0 : self.num_detection_max] for img in eval_imgs], dim=-1
                )[:, inds]
                y_ignore = torch.cat([img["ignore"]["y"] for img in eval_imgs])
                non_ignored = torch.count_nonzero(y_ignore == 0)
                if non_ignored == 0:
                    continue
                # Calculate true positive and false positive based on prediction matches and ignore flags
                tps = torch.logical_and(predm, torch.logical_not(pred_ignore))
                fps = torch.logical_and(torch.logical_not(predm), torch.logical_not(pred_ignore))
                # Calculate precision and recall iteratively
                tp_sum = torch.cumsum(tps, dim=1).to(device=self._device, dtype=torch.float64)
                fp_sum = torch.cumsum(fps, dim=1).to(device=self._device, dtype=torch.float64)

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

                    inds = torch.searchsorted(rc, self.rec_thresholds, right=False)
                    # Find if recording thresholds recalls exist in calculated recalls.
                    # If no such recording thresholds exist, skip computation.
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                    except:
                        pass
                    precision[t, :, c, a] = torch.tensor(q)

        self.precision = precision

    def _summarize(self, iou_thr: Optional[float] = None, area_range: str = "all") -> float:
        area_index = list(self.object_area_ranges.keys()).index(area_range)
        # Calculate Average Precision
        s = self.precision
        if iou_thr is not None:
            t = (self.iou_thresholds == iou_thr).int().nonzero(as_tuple=True)[0]
            s = s[t]
        s = s[:, :, :, area_index]
        # Take mean to calculate mAP
        if len(s[s > -1]) == 0:
            mean_s = torch.tensor(-1)
        else:
            mean_s = torch.mean(s[s > -1])

        return mean_s.item()

    @reinit__is_reduced
    def _gather_all(self) -> None:
        import torch.distributed as dist

        if not dist.is_available():
            return
        if not dist.is_initialized():
            return

        eval_gather_dicts: List[Dict] = [defaultdict(list)] * dist.get_world_size()
        dist.all_gather_object(eval_gather_dicts, self.eval_imgs)

        category_gather_list: List[Set] = [set()] * dist.get_world_size()
        dist.all_gather_object(category_gather_list, self.category_ids)

        combined_eval_imgs: defaultdict = defaultdict(list)
        for eval_imgs in eval_gather_dicts:
            for key, value in eval_imgs.items():
                combined_eval_imgs[key] += value
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

        for area in self.object_area_ranges:
            results[area] = self._summarize(area_range=area)

        return results
