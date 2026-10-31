import math

import pytest
import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import HausdorffDistance, HausdorffDistance95


def test_zero_sample():
    hd = HausdorffDistance()
    with pytest.raises(
        NotComputableError, match=r"HausdorffDistance must have at least one example before it can be computed"
    ):
        hd.compute()


def test_invalid_args():
    with pytest.raises(ValueError, match=r"num_classes must be >= 1"):
        HausdorffDistance(num_classes=0)
    with pytest.raises(ValueError, match=r"percentile must be in"):
        HausdorffDistance(percentile=150.0)
    with pytest.raises(ValueError, match=r"ignore_index can only be used with num_classes > 1"):
        HausdorffDistance(num_classes=1, ignore_index=0)


def test_single_point_pair_2d():
    # Pred has a single foreground point at (0, 0), GT has it at (3, 3) -> HD = sqrt(18) ~ 4.2426
    pred = torch.zeros(1, 4, 4)
    pred[0, 0, 0] = 1
    gt = torch.zeros(1, 4, 4)
    gt[0, 3, 3] = 1

    hd = HausdorffDistance()
    hd.update((pred, gt))
    val = hd.compute()
    assert val == pytest.approx(math.sqrt(18.0), rel=1e-5)


def test_single_point_pair_3d():
    # 3D version: corner-to-corner of a 4x4x4 cube
    pred = torch.zeros(1, 4, 4, 4)
    pred[0, 0, 0, 0] = 1
    gt = torch.zeros(1, 4, 4, 4)
    gt[0, 3, 3, 3] = 1

    hd = HausdorffDistance()
    hd.update((pred, gt))
    val = hd.compute()
    assert val == pytest.approx(math.sqrt(27.0), rel=1e-5)


def test_identical_masks_zero():
    mask = torch.zeros(2, 8, 8)
    mask[:, 2:5, 2:5] = 1
    hd = HausdorffDistance()
    hd.update((mask.clone(), mask))
    assert hd.compute() == pytest.approx(0.0, abs=1e-6)


def test_two_simple_shapes_2d():
    # Two non-overlapping 2x2 squares: top-left vs bottom-right of an 8x8 grid.
    # Boundary points of the squares face each other; the worst-case nearest-neighbor
    # distance between their boundary sets is sqrt((6 - 1)^2 + (6 - 1)^2) = sqrt(50)
    pred = torch.zeros(1, 8, 8)
    pred[0, 0:2, 0:2] = 1
    gt = torch.zeros(1, 8, 8)
    gt[0, 6:8, 6:8] = 1

    hd = HausdorffDistance()
    hd.update((pred, gt))
    val = hd.compute()
    # Each pred point's worst case to gt is the corner farthest from gt -> (0,0) -> (7,7) = sqrt(98)
    # And symmetric. The directed HD from pred-to-gt is achieved at (0,0)->(6,6)=sqrt(72)
    # max over A of min over B; closest gt point to (0,0) is (6,6)=sqrt(72)
    expected = math.sqrt(72.0)
    assert val == pytest.approx(expected, rel=1e-5)


def test_update_lifecycle_average():
    # Two batches, average should be (d1 + d2) / 2
    pred1 = torch.zeros(1, 4, 4)
    pred1[0, 0, 0] = 1
    gt1 = torch.zeros(1, 4, 4)
    gt1[0, 3, 3] = 1
    d1 = math.sqrt(18.0)

    pred2 = torch.zeros(1, 4, 4)
    pred2[0, 0, 0] = 1
    gt2 = torch.zeros(1, 4, 4)
    gt2[0, 0, 2] = 1
    d2 = 2.0

    hd = HausdorffDistance()
    hd.update((pred1, gt1))
    hd.update((pred2, gt2))
    val = hd.compute()
    assert val == pytest.approx((d1 + d2) / 2.0, rel=1e-5)


def test_reset():
    pred = torch.zeros(1, 4, 4)
    pred[0, 0, 0] = 1
    gt = torch.zeros(1, 4, 4)
    gt[0, 3, 3] = 1

    hd = HausdorffDistance()
    hd.update((pred, gt))
    hd.reset()
    with pytest.raises(NotComputableError):
        hd.compute()


def test_hd95_is_smaller_or_equal():
    # Construct a noisy mask where a single outlier inflates HD; HD95 should be strictly smaller.
    pred = torch.zeros(1, 16, 16)
    pred[0, 0:4, 0:4] = 1  # main blob
    pred[0, 15, 15] = 1  # outlier far away
    gt = torch.zeros(1, 16, 16)
    gt[0, 0:4, 0:4] = 1  # matches main blob

    hd = HausdorffDistance()
    hd.update((pred, gt))
    full_hd = hd.compute()

    hd95 = HausdorffDistance95()
    hd95.update((pred, gt))
    p95_hd = hd95.compute()

    assert p95_hd < full_hd
    # the outlier at (15, 15) is farthest from any GT boundary point; nearest GT boundary is (3, 3)
    # so distance = sqrt((15-3)^2 + (15-3)^2) = sqrt(288)
    assert full_hd == pytest.approx(math.sqrt(288.0), rel=1e-5)


def test_multiclass_2d():
    # Multi-class with num_classes=3. Provide y_pred as one-hot logits.
    # Class 1: pred and gt identical at (0,0). Class 2: pred at (0,0), gt at (3,3).
    # Per-class distances: 0.0 and sqrt(18). Mean = sqrt(18)/2.
    H, W = 4, 4
    y_pred_logits = torch.zeros(1, 3, H, W)
    # default high logit for background class
    y_pred_logits[0, 0, :, :] = 1.0
    # class 1 pixel at (0, 0)
    y_pred_logits[0, 0, 0, 0] = 0.0
    y_pred_logits[0, 1, 0, 0] = 5.0
    # class 2 pixel at (0, 0)
    y_pred_logits[0, 0, 0, 1] = 0.0
    y_pred_logits[0, 2, 0, 1] = 5.0

    y = torch.zeros(1, H, W, dtype=torch.long)
    y[0, 0, 0] = 1  # class 1 matches
    y[0, 3, 3] = 2  # class 2 ground truth different from pred

    hd = HausdorffDistance(num_classes=3, ignore_index=0)
    hd.update((y_pred_logits, y))
    val = hd.compute()
    # class 1 hd = 0 (pred and gt both at (0,0))
    # class 2 hd: pred set {(0,1)}, gt set {(3,3)} -> sqrt((3-0)^2 + (3-1)^2) = sqrt(13)
    expected = (0.0 + math.sqrt(13.0)) / 2.0
    assert val == pytest.approx(expected, rel=1e-5)


def test_multiclass_ignore_index():
    H, W = 4, 4
    y_pred_logits = torch.zeros(1, 3, H, W)
    y_pred_logits[0, 0, :, :] = 1.0  # background dominant
    y_pred_logits[0, 0, 0, 0] = 0.0
    y_pred_logits[0, 2, 0, 0] = 5.0  # pred class 2 at (0,0)

    y = torch.zeros(1, H, W, dtype=torch.long)
    y[0, 3, 3] = 2  # gt class 2 at (3,3)

    # ignore class 0 (background) and class 1 (no positives anywhere) — only class 2 remains
    hd = HausdorffDistance(num_classes=3, ignore_index=0)
    hd.update((y_pred_logits, y))
    # class 1 has empty pred and empty gt -> 0; class 2 -> sqrt(18) -> mean = sqrt(18)/2
    expected = (0.0 + math.sqrt(18.0)) / 2.0
    assert hd.compute() == pytest.approx(expected, rel=1e-5)


def test_invalid_input_rank():
    pred = torch.zeros(4, 4)
    gt = torch.zeros(4, 4)
    hd = HausdorffDistance()
    with pytest.raises(ValueError, match=r"Expected inputs of rank"):
        hd.update((pred, gt))


def test_with_engine():
    from ignite.engine import Engine

    pred1 = torch.zeros(1, 4, 4)
    pred1[0, 0, 0] = 1
    gt1 = torch.zeros(1, 4, 4)
    gt1[0, 3, 3] = 1

    data = [(pred1, gt1)]

    def step(_engine, batch):
        return batch

    engine = Engine(step)
    hd = HausdorffDistance()
    hd.attach(engine, "hd")
    state = engine.run(data, max_epochs=1)
    assert "hd" in state.metrics
    assert state.metrics["hd"] == pytest.approx(math.sqrt(18.0), rel=1e-5)
