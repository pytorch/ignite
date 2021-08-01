from collections import defaultdict

import torch

from ignite.metrics.metric import Metric


def iou(gt, dt, crowd):
    m = len(dt)
    n = len(gt)

    ious = torch.zeros(m, n)

    for g in range(n):
        G = gt[g]
        ga = G[2] * G[3]
        iscrowd = crowd[g]
        for d in range(m):
            D = dt[d]
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


class AP(Metric):
    def __init__(self, iou_thrs, rec_thrs, area_rngs, max_dets, device):
        self._gts = defaultdict(lambda: torch.zeros(0, 11))
        self._dts = defaultdict(lambda: torch.zeros(0, 10))

        self.area_rngs = area_rngs
        self.max_dets = sorted(max_dets)

        self.iou_thrs = iou_thrs
        self.rec_thrs = rec_thrs
        self.device = device

        self.evalImgs = defaultdict(list)
        self.eval = {}

        self.stats = []
        self.ious = {}

        self.class_ids = []
        self.img_ids = []

    def update(self, dts, gts):
        # Accumulate ground truths in gt dictionary
        for gt in gts:
            img_id = int(gt[1])
            class_id = int(gt[2])
            self._gts[img_id, class_id] = torch.vstack([self._gts[img_id, class_id], gt])
            if img_id not in self.img_ids:
                self.img_ids.append(img_id)
            if class_id not in self.class_ids:
                self.class_ids.append(class_id)

        # Accumulate predictions in pred dictionary
        for dt in dts:
            img_id = int(dt[1])
            class_id = int(dt[2])
            self._dts[img_id, class_id] = torch.vstack([self._dts[img_id, class_id], dt])
            if img_id not in self.img_ids:
                self.img_ids.append(img_id)
            if class_id not in self.class_ids:
                self.class_ids.append(class_id)

    def compute_iou(self, img_id, class_id):
        gt = self._gts[img_id, class_id]
        dt = self._dts[img_id, class_id]

        dt = dt[torch.argsort(dt[:, 9])]

        crowd = [g[8] for g in gt]
        ious = iou(gt[:, 3:7], dt[:, 3:7], crowd)

        return ious

    def evaluate_image(self, img_id, class_id, area_rng, max_det):
        # Get all ground truths and predictions for given image id and class is
        gt = self._gts[img_id, class_id]
        dt = self._dts[img_id, class_id]

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Set ignore flag for all ground truths with area range less than or greater than given area range
        for g in gt:
            if g[9] or (g[7] < area_rng[0] or g[7] > area_rng[1]):
                g[10] = 1
            else:
                g[10] = 0

        # Sort ground truths so that not ignored ground truths are at front
        gt_ind = torch.argsort(gt[:, 10])
        gt = gt[gt_ind]

        # Sort predictions based on confidence/score
        dt = dt[torch.argsort(-dt[:, 9])]
        # iscrowd flag to show multiple ground truths
        iscrowd = gt[:, 8]

        ious = (
            self.ious[img_id, class_id][:, gt_ind]
            if len(self.ious[img_id, class_id]) > 0
            else self.ious[img_id, class_id]
        )

        num_iou_thrs = len(self.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)
        gtm = torch.zeros(num_iou_thrs, num_gt)
        dtm = torch.zeros(num_iou_thrs, num_dt)
        gt_ignore = gt[:, 10]
        dt_ignore = torch.zeros(num_iou_thrs, num_dt)

        if len(ious) != 0:
            for tind, t in enumerate(self.iou_thrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # If ground truth is already matched with another prediction and not crowd
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue

                        # if prediction matched to reg ground truth, and on ignore ground truth, stop
                        if m > -1 and gt_ignore[m] == 0 and gt_ignore[gind] == 1:
                            break

                        # continue to next ground truth unless better match made
                        if ious[dind, gind] < iou:
                            continue

                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind

                    # if match made store id of match for both prediction and ground truth
                    if m == -1:
                        continue
                    dt_ignore[tind, dind] = gt_ignore[m]
                    dtm[tind, dind] = gt[m][0]
                    gtm[tind, m] = d[0]

        # set unmatched predictions outside of area range to ignore
        a = torch.tensor([d[7] < area_rng[0] or d[7] > area_rng[1] for d in dt]).reshape((1, len(dt)))

        dt_ignore = torch.logical_or(
            dt_ignore, torch.logical_and(dtm == 0, torch.repeat_interleave(a, num_iou_thrs, 0))
        )

        return {
            "image_id": img_id,
            "category_id": class_id,
            "aRng": area_rng,
            "maxDet": max_det,
            "dtIds": dt[:, 0],
            "gtIds": gt[:, 0],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": dt[:, 9],
            "gtIgnore": gt_ignore,
            "dtIgnore": dt_ignore,
        }

    def accumulate(self):
        num_iou_thr = len(self.iou_thrs)
        num_rec_thr = len(self.rec_thrs)
        num_classes = len(self.class_ids)
        num_area = len(self.area_rngs)
        num_dets = len(self.max_dets)

        precision = -torch.ones((num_iou_thr, num_rec_thr, num_classes, num_area, num_dets))
        recall = -torch.ones((num_iou_thr, num_classes, num_area, num_dets))
        scores = -torch.ones((num_iou_thr, num_rec_thr, num_classes, num_area, num_dets))

        num_imgs = len(self.img_ids)
        # retrieve eval_imgs at each category, area range, and max number of detections
        for c, class_id in enumerate(self.class_ids):
            num_class_c = c * num_area * num_imgs
            for a, area_rng in enumerate(self.area_rngs):
                num_area_a = a * num_imgs
                for m, max_det in enumerate(self.max_dets):
                    # retrieve appropriate eval_imgs from stored results
                    eval_imgs = [self.eval_imgs[num_class_c + num_area_a + i] for i in range(num_imgs)]
                    eval_imgs = [img for img in eval_imgs if img is not None]
                    if len(eval_imgs) == 0:
                        continue
                    # Get prediction scores to greedily match
                    pred_scores = torch.cat([img["dtScores"][0:max_det] for img in eval_imgs], dim=-1)
                    # Sort prediction scores
                    inds = torch.argsort(-pred_scores)
                    sorted_pred_scores = pred_scores[inds]
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
                        tp = torch.tensor(tp)
                        fp = torch.tensor(fp)
                        nd = len(tp)
                        rc = tp / non_ignored
                        pr = tp / (fp + tp + torch.finfo(torch.float64).eps)
                        q = torch.zeros((num_rec_thr,))
                        ss = torch.zeros((num_rec_thr,))

                        if nd:
                            recall[t, c, a, m] = rc[-1]
                        else:
                            recall[t, c, a, m] = 0

                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = torch.searchsorted(rc, self.rec_thrs, right=False)
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = sorted_pred_scores[pi]
                        except:
                            pass
                        precision[t, :, c, a, m] = torch.tensor(q)
                        scores[t, :, c, a, m] = torch.tensor(ss)
        # return precision, recall
        self.eval = {
            "precision": precision,
            "recall": recall,
        }

    def _summarize(self, ap=1, iou_thr=None, area_rng="all", max_det=100):
        aind = [i for i, a_rng in enumerate(self.area_rngs) if a_rng[2] == area_rng]
        mind = [i for i, m_det in enumerate(self.max_dets) if m_det == max_det]
        # Calculate Average Precision
        if ap == 1:
            s = self.eval["precision"]
            if iou_thr is not None:
                t = (self.iou_thrs == iou_thr).int().nonzero(as_tuple=True)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        # Calculate Average Recall
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                t = (self.iou_thrs == iou_thr).nonzero(as_tuple=True)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        # Take mean to calculate mAP or mAR
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = torch.mean(s[s > -1])

        return mean_s

    def summarize_all(self):
        stats = torch.zeros((12,))
        stats[0] = self._summarize(1)
        stats[1] = self._summarize(1, iou_thr=0.5, max_det=self.max_dets[2])
        stats[2] = self._summarize(1, iou_thr=0.75, max_det=self.max_dets[2])
        stats[3] = self._summarize(1, area_rng="small", max_det=self.max_dets[2])
        stats[4] = self._summarize(1, area_rng="medium", max_det=self.max_dets[2])
        stats[5] = self._summarize(1, area_rng="large", max_det=self.max_dets[2])
        stats[6] = self._summarize(0, max_det=self.max_dets[0])
        stats[7] = self._summarize(0, max_det=self.max_dets[1])
        stats[8] = self._summarize(0, max_det=self.max_dets[2])
        stats[9] = self._summarize(0, area_rng="small", max_det=self.max_dets[2])
        stats[10] = self._summarize(0, area_rng="medium", max_det=self.max_dets[2])
        stats[11] = self._summarize(0, area_rng="large", max_det=self.max_dets[2])

        return stats

    def compute(self):
        self.class_ids.sort()
        self.img_ids.sort()
        self.ious = {
            (img_id, class_id): self.compute_iou(img_id, class_id)
            for img_id in self.img_ids
            for class_id in self.class_ids
        }

        max_det = self.max_dets[-1]
        self.eval_imgs = [
            self.evaluate_image(img_id, class_id, area_rng, max_det)
            for class_id in self.class_ids
            for area_rng in self.area_rngs
            for img_id in self.img_ids
        ]

        self.accumulate()

        results = self.summarize_all()

        return results
