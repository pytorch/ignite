import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ignite.metrics import MeanAveragePrecision

torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)


def get_coco_val2017_3samples():
    gt = [
        torch.tensor(
            [
                [126, 90, 523, 534, 6],
                [190, 304, 249, 369, 1],
                [435, 338, 451, 362, 1],
                [298, 334, 332, 367, 1],
                [174, 170, 203, 192, 1],
                [297, 160, 322, 180, 1],
                [121, 389, 127, 410, 1],
                [568, 316, 611, 404, 6],
                [91, 388, 104, 422, 1],
                [212, 168, 230, 188, 1],
                [78, 377, 97, 429, 1],
                [101, 397, 114, 429, 1],
                [113, 391, 126, 429, 1],
                [502, 315, 565, 384, 6],
            ]
        ),
        torch.tensor(
            [
                [256, 70, 353, 302, 1],
                [87, 184, 240, 324, 1],
                [87, 71, 153, 140, 1],
                [169, 74, 221, 130, 1],
                [387, 60, 475, 113, 1],
                [215, 76, 262, 129, 1],
                [301, 28, 363, 97, 39],
                [39, 0, 75, 14, 1],
                [138, 115, 191, 133, 15],
                [340, 97, 418, 118, 15],
                [213, 230, 241, 263, 40],
                [147, 103, 152, 118, 44],
                [17, 96, 67, 148, 1],
                [49, 80, 102, 142, 1],
                [86, 107, 90, 123, 44],
                [16, 141, 126, 323, 1],
            ]
        ),
        torch.tensor(
            [
                [180, 281, 218, 326, 18],
                [51, 12, 599, 399, 4],
                [221, 100, 289, 165, 2],
                [218, 158, 279, 213, 2],
                [398, 144, 519, 234, 8],
            ]
        ),
    ]

    pred = [
        torch.tensor(
            [
                [124, 90, 522, 534, 1, 6],
                [507, 316, 567, 385, 1, 6],
                [76, 375, 97, 427, 0.99, 1],
                [189, 307, 250, 368, 0.98, 1],
                [113, 383, 128, 430, 0.98, 1],
                [208, 21, 226, 43, 0.92, 18],
                [208, 21, 226, 43, 0.92, 18],
                [208, 21, 226, 43, 0.92, 16],
                [208, 21, 226, 43, 0.92, 16],
                [208, 21, 226, 43, 0.92, 16],
                [208, 21, 226, 43, 0.92, 16],
                [208, 21, 226, 43, 0.92, 16],
                [208, 21, 226, 44, 0.92, 16],
                [208, 21, 226, 43, 0.92, 16],
                [208, 21, 226, 43, 0.92, 16],
                [99, 397, 115, 429, 0.91, 1],
                [176, 166, 201, 192, 0.88, 1],
                [570, 320, 612, 404, 0.86, 6],
                [91, 390, 103, 430, 0.84, 1],
                [397, 178, 411, 198, 0.79, 1],
                [569, 339, 610, 403, 0.65, 3],
                [61, 354, 94, 368, 0.56, 3],
                [177, 177, 198, 192, 0.38, 77],
            ]
        ),
        torch.tensor(
            [
                [262, 71, 350, 297, 1, 1],
                [94, 185, 233, 320, 1, 1],
                [17, 139, 126, 324, 1, 1],
                [214, 231, 241, 263, 1, 40],
                [397, 60, 452, 114, 0.99, 1],
                [167, 74, 221, 133, 0.99, 1],
                [89, 72, 149, 140, 0.99, 1],
                [300, 28, 363, 98, 0.99, 39],
                [212, 72, 262, 129, 0.99, 1],
                [24, 94, 71, 147, 0.98, 1],
                [48, 81, 77, 137, 0.96, 1],
                [285, 91, 303, 119, 0.92, 40],
                [85, 106, 91, 124, 0.89, 44],
                [145, 102, 151, 119, 0.87, 44],
                [29, 0, 78, 14, 0.68, 1],
                [0, 92, 22, 146, 0.61, 1],
                [50, 156, 195, 323, 0.56, 1],
                [13, 0, 56, 15, 0.53, 1],
                [83, 0, 109, 12, 0.50, 1],
                [136, 116, 179, 135, 0.43, 15],
                [261, 80, 299, 131, 0.40, 1],
                [250, 105, 263, 121, 0.31, 40],
                [111, 118, 155, 140, 0.30, 31],
                [12, 87, 54, 146, 0.29, 1],
            ]
        ),
        torch.tensor(
            [
                [145, 15, 564, 385, 0.82, 4],
                [406, 137, 524, 230, 0.75, 3],
                [178, 281, 219, 325, 0.74, 18],
                [145, 185, 153, 204, 0.73, 44],
                [152, 186, 160, 205, 0.70, 44],
                [413, 149, 569, 237, 0.61, 8],
                [352, 116, 550, 240, 0.54, 8],
                [167, 189, 175, 208, 0.53, 44],
                [160, 185, 168, 206, 0.51, 44],
                [216, 160, 280, 211, 0.47, 2],
                [219, 108, 283, 163, 0.47, 2],
                [133, 201, 294, 327, 0.36, 4],
                [222, 145, 582, 391, 0.35, 2],
                [127, 217, 277, 329, 0.32, 15],
            ]
        ),
    ]

    return gt, pred


def get_gt_partially_empty_3samples():
    gt, pred = get_coco_val2017_3samples()
    gt[0] = torch.Tensor(0, 5)
    return gt, pred


def get_pred_partially_empty_3samples():
    gt, pred = get_coco_val2017_3samples()
    pred[1] = torch.Tensor(0, 6)
    return gt, pred


def get_both_partially_empty_3samples():
    gt, pred = get_coco_val2017_3samples()
    gt[0] = torch.Tensor(0, 5)
    gt[2] = torch.Tensor(0, 5)
    pred[1] = torch.Tensor(0, 6)
    return gt, pred


def get_both_partially_empty_3samples_2():
    gt, pred = get_coco_val2017_3samples()
    gt[1] = torch.Tensor(0, 5)
    pred[1] = torch.Tensor(0, 6)
    return gt, pred


def get_all_empty_3samples():
    return [torch.Tensor(0, 5), torch.Tensor(0, 6)]


def create_coco_api(predictions, targets):
    """Create COCO object from predictions and targets

    Args:
        predictions torch.Tensor: predictions in (N, 5) shape where 5 is (x1, y1, x2, y2, score, class_id)
        targets torch.Tensor: targets in (N, 6) shape where 6 is (x1, y1, x2, y2, class_id)

    Returns:
        Tuple[coco_api.COCO, coco_api.COCO]: coco object
    """
    ann_id = 1
    coco_gt = COCO()
    dataset = {"images": [], "categories": [], "annotations": []}

    for idx, target in enumerate(targets):
        dataset["images"].append({"id": idx})
        for i in range(target.shape[0]):
            bbox = target[i][:4]
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            bbox = [x.item() for x in bbox]
            area = bbox[2] * bbox[3]
            ann = {
                "image_id": idx,
                "bbox": bbox,
                "category_id": target[i][4].item(),
                "area": area,
                "iscrowd": False,
                "id": ann_id,
            }
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in range(0, 100)]
    coco_gt.dataset = dataset
    coco_gt.createIndex()

    for idx, prediction in enumerate(predictions):
        prediction[:, 2:4] = prediction[:, 2:4] - prediction[:, 0:2]
        predictions[idx] = torch.cat([torch.tensor(idx).repeat(prediction.shape[0], 1), prediction], dim=1)
    predictions = torch.cat(predictions, dim=0)
    coco_dt = coco_gt.loadRes(predictions.numpy())
    return coco_gt, coco_dt


def _test_compute(predictions, targets, device, approx=1e-2):
    coco_gt, coco_dt = create_coco_api(
        [torch.clone(pred) for pred in predictions], [torch.clone(target) for target in targets]
    )
    eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    metric_50 = MeanAveragePrecision(iou_thresholds=[0.5], device=device)
    metric_75 = MeanAveragePrecision(iou_thresholds=[0.75], device=device)
    metric_50_95 = MeanAveragePrecision(device=device)

    targets = [t.to(device) for t in targets]
    predictions = [p.to(device) for p in predictions]

    for prediction, target in zip(predictions, targets):
        metric_50.update((prediction, target))
        metric_75.update((prediction, target))
        metric_50_95.update((prediction, target))

    res_50 = metric_50.compute()
    res_75 = metric_75.compute()
    res_50_95 = metric_50_95.compute()

    assert eval.stats[0] == pytest.approx(res_50_95, abs=approx)
    assert eval.stats[1] == pytest.approx(res_50, abs=approx)
    assert eval.stats[2] == pytest.approx(res_75, abs=approx)


def test_gt_pred_all_exist():
    targets, predictions = get_coco_val2017_3samples()
    _test_compute(predictions, targets, torch.device("cpu"))
    if torch.cuda.is_available():
        _test_compute(predictions, targets, torch.device("cuda"), approx=2e-2)


def test_gt_partially_empty():
    targets, predictions = get_gt_partially_empty_3samples()
    _test_compute(predictions, targets, torch.device("cpu"))
    if torch.cuda.is_available():
        _test_compute(predictions, targets, torch.device("cuda"), approx=2e-2)


def test_pred_partially_empty():
    targets, predictions = get_pred_partially_empty_3samples()
    _test_compute(predictions, targets, torch.device("cpu"))
    if torch.cuda.is_available():
        _test_compute(predictions, targets, torch.device("cuda"), approx=2e-2)


def test_both_partially_empty():
    targets, predictions = get_both_partially_empty_3samples()
    _test_compute(predictions, targets, torch.device("cpu"))
    if torch.cuda.is_available():
        _test_compute(predictions, targets, torch.device("cuda"), approx=2e-2)


def test_both_partially_empty_2():
    targets, predictions = get_both_partially_empty_3samples_2()
    _test_compute(predictions, targets, torch.device("cpu"))
    if torch.cuda.is_available():
        _test_compute(predictions, targets, torch.device("cuda"), approx=2e-2)


def test_no_torchvision():
    with patch.dict(sys.modules, {"torchvision.ops": None}):
        with pytest.raises(ModuleNotFoundError, match=r"This module requires torchvision to be installed."):
            MeanAveragePrecision()
