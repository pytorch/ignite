import pytest
import torch
from torch.testing import assert_close

import ignite.distributed as idist
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score
from ignite.exceptions import NotComputableError
from ignite.metrics.fairness.accuracy_difference import SubgroupAccuracyDifference


def test_subgroup_accuracy_difference_empty() -> None:
    """Tests if NotComputableError is raised when no data is provided."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])
    with pytest.raises(NotComputableError, match="Fairness metrics must have at least one example"):
        metric.compute()


def test_subgroup_accuracy_difference_single_group() -> None:
    """Tests if NotComputableError is raised when only one subgroup is present."""
    metric = SubgroupAccuracyDifference(groups=[0])

    y_pred = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    y = torch.tensor([0, 0])
    group_labels = torch.tensor([0, 0])

    metric.update((y_pred, y, group_labels))

    with pytest.raises(NotComputableError, match="Fairness metrics require at least two unique subgroups"):
        metric.compute()


def test_subgroup_accuracy_difference_binary_labels(available_device) -> None:
    """Tests SubgroupAccuracyDifference with binary 0/1 labels."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], device=available_device)

    # y_pred and y are (B,)
    # Group 0: 2/2 correct (1.0)
    # Group 1: 0/2 correct (0.0)
    y_pred = torch.tensor([1, 0, 1, 0], device=available_device)
    y = torch.tensor([1, 0, 0, 1], device=available_device)
    groups = torch.tensor([0, 0, 1, 1], device=available_device)

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def test_subgroup_accuracy_difference_binary_probs(available_device) -> None:
    """Tests SubgroupAccuracyDifference with (B, 1) thresholded labels."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], device=available_device)

    # y_pred is (B, 1), y is (B,)
    # Group 0: 2/2 correct (1.0)
    # Group 1: 1/2 correct (0.5)
    y_pred = torch.tensor([[1], [0], [1], [1]], device=available_device)
    y = torch.tensor([1, 0, 1, 0], device=available_device)
    groups = torch.tensor([0, 0, 1, 1], device=available_device)

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 0.5)


def test_subgroup_accuracy_difference_multiclass(available_device) -> None:
    """Tests SubgroupAccuracyDifference with multiclass logits."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], device=available_device)

    # y_pred is (B, C), y is (B,)
    # Group 0: 1/1 correct (1.0)
    # Group 1: 0/1 correct (0.0)
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], device=available_device)
    y = torch.tensor([1, 1], device=available_device)
    groups = torch.tensor([0, 1], device=available_device)

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def test_subgroup_accuracy_difference_multilabel(available_device) -> None:
    """Tests SubgroupAccuracyDifference with multilabel data."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], is_multilabel=True, device=available_device)

    # y_pred and y are (B, C)
    # Accuracy uses sample-wise correctness: all labels must match per sample.
    # Group 0 (sample 0): y_pred=[1,0], y=[1,0] -> all match -> correct. Accuracy = 1/1 = 1.0
    # Group 1 (sample 1): y_pred=[1,1], y=[1,0] -> not all match -> incorrect. Accuracy = 0/1 = 0.0
    y_pred = torch.tensor([[1, 0], [1, 1]], device=available_device)
    y = torch.tensor([[1, 0], [1, 0]], device=available_device)
    groups = torch.tensor([0, 1], device=available_device)

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def test_subgroup_accuracy_difference_spatial(available_device) -> None:
    """Tests SubgroupAccuracyDifference with spatial (image) data."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], device=available_device)

    # y_pred is (B, C, H, W), y is (B, H, W)
    # B=2, C=2, H=2, W=2
    # Group 0: spatial targets (4 pixels) all correct (1.0)
    # Group 1: spatial targets (4 pixels) all incorrect (0.0)

    y_pred = torch.zeros(2, 2, 2, 2, device=available_device)
    # Group 0 (index 0): predict class 0 for all
    y_pred[0, 0, :, :] = 1.0
    # Group 1 (index 1): predict class 0 for all
    y_pred[1, 0, :, :] = 1.0

    y = torch.zeros(2, 2, 2, dtype=torch.long, device=available_device)
    # Group 0: all class 0 (1.0 acc)
    y[0, :, :] = 0
    # Group 1: all class 1 (0.0 acc)
    y[1, :, :] = 1

    groups = torch.tensor([0, 1], device=available_device)

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 1.0)


def test_compare_accuracy_difference_with_fairlearn(available_device) -> None:
    """Verifies SubgroupAccuracyDifference matches Fairlearn's MetricFrame(accuracy_score).difference()"""
    groups_list = [0, 1]
    ignite_metric = SubgroupAccuracyDifference(groups=groups_list, device=available_device)

    # Random binary data
    y_pred_probs = torch.tensor(
        [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]], device=available_device
    )
    y_pred = torch.argmax(y_pred_probs, dim=1)
    y_true = torch.tensor([1, 0, 1, 1, 0, 0], device=available_device)
    group_labels = torch.tensor([0, 0, 0, 1, 1, 1], device=available_device)

    # Ignite update and compute
    ignite_metric.update((y_pred_probs, y_true, group_labels))
    ignite_result = ignite_metric.compute()

    # Fairlearn computation
    # Fairlearn's MetricFrame takes numpy arrays
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true.cpu().numpy(),
        y_pred=y_pred.cpu().numpy(),
        sensitive_features=group_labels.cpu().numpy(),
    )
    fairlearn_result = mf.difference()

    assert_close(ignite_result, float(fairlearn_result))


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_compute_subgroup_accuracy_difference(self) -> None:
        rank = idist.get_rank()
        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            metric = SubgroupAccuracyDifference(groups=[0, 1], device=metric_device)

            y = torch.tensor([0, 1], device=device)
            groups = torch.tensor([0, 1], device=device)

            if rank == 0:
                # Group 0: correct, Group 1: incorrect
                y_pred = torch.tensor([[0.8, 0.2], [0.8, 0.2]], device=device)
            else:
                # Group 0: correct, Group 1: incorrect
                y_pred = torch.tensor([[0.9, 0.1], [0.9, 0.1]], device=device)

            metric.update((y_pred, y, groups))
            res = metric.compute()

            # Across all ranks:
            # Group 0: all correct (1.0)
            # Group 1: all incorrect (0.0)
            assert_close(res, 1.0)


def test_subgroup_accuracy_difference_validation() -> None:
    """Tests input validation for SubgroupAccuracyDifference."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # Valid multiclass logits
    y_pred = torch.randn(4, 3)
    y = torch.randint(0, 3, (4,))
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    assert metric.compute() >= 0

    metric.reset()

    # Valid binary classes (0/1)
    y_pred = torch.tensor([0, 1, 0, 1])
    y = torch.tensor([0, 1, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    assert metric.compute() == 1.0  # Subgroup 0: 100%, Subgroup 1: 0%

    metric.reset()

    # Invalid binary probabilities (not 0 or 1)
    y_pred = torch.tensor([0.6, 0.4, 0.8, 0.1])
    y = torch.tensor([1, 0, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="y_pred must be comprised of 0's and 1's"):
        metric.update((y_pred, y, groups))


def test_multilabel_validation() -> None:
    """Tests input validation for multilabel data."""
    metric = SubgroupAccuracyDifference(groups=[0, 1], is_multilabel=True)

    # Valid multilabel (0/1)
    y_pred = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1]])
    y = torch.tensor([[1, 0], [1, 1], [1, 1], [0, 0]])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    # Accuracy uses sample-wise correctness: all labels must match per sample.
    # Group 0 (samples 0,1): [1,0]==[1,0], [1,1]==[1,1] -> Accuracy = 2/2 = 1.0
    # Group 1 (samples 2,3): [0,0]!=[1,1], [0,1]!=[0,0] -> Accuracy = 0/2 = 0.0
    # Disparity = 1.0 - 0.0 = 1.0
    assert metric.compute() == 1.0

    metric.reset()

    # Invalid multilabel (not 0/1)
    y_pred = torch.tensor([[0.6, 0.4], [0.8, 0.1]])
    y = torch.tensor([[1, 0], [1, 1]])
    groups = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="y_pred must be comprised of 0's and 1's"):
        metric.update((y_pred, y, groups))


def test_shape_mismatch_validation() -> None:
    """Tests validation for shape mismatches between y and y_pred."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y is (B,), y_pred is (B, C) - OK
    metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(4)))

    # y is (B,), y_pred is (B, C, H, W) - Error (dimension mismatch)
    with pytest.raises(ValueError, match="y must have shape of"):
        metric.update((torch.randn(4, 3, 2, 2), torch.randint(0, 3, (4,)), torch.zeros(4)))


def test_batch_size_mismatch_validation() -> None:
    """Tests validation for batch size mismatches across y_pred, y, and groups."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # y_pred (4), y (3) - Batch size mismatch
    with pytest.raises(ValueError, match="y_pred, y, and group_labels must have the same batch size"):
        metric.update((torch.randn(4, 3), torch.randint(0, 3, (3,)), torch.zeros(4)))

    # groups (3), y_pred (4) - Batch size mismatch
    with pytest.raises(ValueError, match="y_pred, y, and group_labels must have the same batch size"):
        metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(3)))


def test_type_switch_validation() -> None:
    """Tests if RuntimeError is raised when input type changes mid-epoch."""
    metric = SubgroupAccuracyDifference(groups=[0, 1])

    # First batch: binary
    metric.update((torch.tensor([1, 0]), torch.tensor([1, 0]), torch.tensor([0, 1])))

    # Second batch: multiclass - Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Input data type has changed from binary to multiclass"):
        metric.update((torch.randn(2, 3), torch.tensor([0, 1]), torch.tensor([0, 1])))

    # Third batch: multiclass (3) then multiclass (5) - Should raise ValueError
    metric.reset()
    metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(4)))
    with pytest.raises(ValueError, match="Input data number of classes has changed"):
        metric.update((torch.randn(4, 5), torch.randint(0, 5, (4,)), torch.zeros(4)))
