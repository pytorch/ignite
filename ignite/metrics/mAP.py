from collections import defaultdict

import torch

from ignite.metrics.metric import Metric


def group_detections(gts, preds):
    bbs = defaultdict(lambda: {"pred": [], "gt": []})

    for gt in gts:
        bbs[gt[1], gt[2]]["gt"].append(gt)
    for pred in preds:
        bbs[pred[1], pred[2]]["pred"].append(pred)

    return bbs


def intersection_over_union(a, b):
    xa, ya, x2a, y2a = a[0], a[1], a[2], a[3]
    xb, yb, x2b, y2b = b[0], b[1], b[2], b[3]

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)

    return Ai / (Aa + Ab - Ai)


def compute_ious(preds, gts):
    ious = torch.zeros((len(preds), len(gts)))
    for g_idx, gt in enumerate(gts):
        for pred_idx, pred in enumerate(preds):
            ious[pred_idx, g_idx] = intersection_over_union(pred[3:7], gt[3:7])

    return ious


def evaluate_img(preds, gts, ious, iou_threshold, max_dets=100, area_rng=[0, 500]):
    print(ious)
    arg_pred = torch.argsort(preds[:, 8])

    preds = preds[arg_pred]
    preds = preds[:max_dets]
    ious = ious[arg_pred]
    ious = ious[:max_dets]

    gt_ignore = torch.logical_not(torch.logical_and(area_rng[0] <= gts[:, 7], gts[:, 7] <= area_rng[1]))
    arg_gts = torch.argsort(gt_ignore)
    gts = gts[arg_gts]
    gt_ignore = gt_ignore[arg_gts]
    ious = ious[:, arg_gts]

    gtm = {}
    predm = {}

    for pred_idx, pred in enumerate(preds):
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, gt in enumerate(gts):
            if g_idx in gtm:
                continue
            if m > -1 and gt_ignore[m] is False and gt_ignore[g_idx] is True:
                break
            if ious[pred_idx, g_idx] < iou:
                continue
            iou = ious[pred_idx, g_idx]
            m = g_idx
        if m == -1:
            continue
        predm[pred_idx] = m
        gtm[m] = pred_idx

    pred_ignore = [
        gt_ignore[predm[pred_idx]] if pred_idx in predm else not (area_rng[0] <= pred[7] and pred[7] <= area_rng[1])
        for pred_idx, pred in enumerate(preds)
    ]

    scores = torch.tensor([preds[pred_idx][8] for pred_idx in range(len(preds)) if not pred_ignore[pred_idx]])
    matched = torch.tensor([pred_idx in predm for pred_idx in range(len(preds)) if not pred_ignore[pred_idx]])

    n_gts = len([g_idx for g_idx in range(len(gts)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "NP": n_gts}


def accumulate_maximum(arr):
    n = len(arr)

    max_arr = torch.zeros(n)
    max_so_far = arr[0]

    for i in range(n):
        if arr[i] > max_so_far:
            max_so_far = arr[i]

        max_arr[i] = max_so_far

    return max_arr


def compute_ap_recall(scores, matched, NP, recall_thresholds=None):

    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
        }

    if recall_thresholds is None:
        recall_thresholds = torch.linspace(0.0, 1.00, int(round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)

    inds = torch.argsort(-scores)

    scores = scores[inds]
    matched = matched[inds]

    tp = torch.cumsum(matched)
    fp = torch.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    i_pr = accumulate_maximum(pr[::-1])[::-1]

    rec_idx = torch.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    i_pr = torch.tensor([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": torch.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
    }


class AP(Metric):
    def __init__(self):
        super(AP, self).__init__()

        self.gts = torch.zeros(0, 5)
        self.preds = torch.zeros(0, 6)

    def update(self, gts, preds):
        self.gts = torch.cat(self.gts, gts)
        self.preds = torch.cat(self.preds, preds)

    def compute(self):
        bbs = group_detections(self.gts, self.preds)

        ious = {k: compute_ious(v["pred"], v["gt"]) for k, v in bbs.items()}

        def evaluate(iou_threshold, max_dets, area_range):
            evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
            for img_id, class_id in bbs:
                ev = evaluate_img(
                    bbs[img_id, class_id]["dt"],
                    bbs[img_id, class_id]["gt"],
                    ious[img_id, class_id],
                    iou_threshold,
                    max_dets,
                    area_range,
                )
                acc = evals[class_id]
                acc["scores"].append(ev["scores"])
                acc["matched"].append(ev["matched"])
                acc["NP"].append(ev["NP"])

            for class_id in evals:
                acc = evals[class_id]
                acc["scores"] = torch.cat(acc["scores"])
                acc["matched"] = torch.cat(acc["matched"]).astype(torch.bool)
                acc["NP"] = torch.sum(acc["NP"])

            res = []
            for class_id in evals:
                ev = evals[class_id]
                eval_results = {
                    "class": class_id,
                }
                for (k, v) in compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]).items():
                    eval_results[k] = v
                res.append(eval_results)
            return res

        iou_thresholds = torch.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05)) + 1)

        full = {i: evaluate(iou_threshold=i, max_dets=100, area_range=(0, float("inf"))) for i in iou_thresholds}
