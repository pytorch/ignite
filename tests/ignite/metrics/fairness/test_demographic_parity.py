import pytest
import torch
from torch.testing import assert_close

import ignite.distributed as idist
from fairlearn.metrics import demographic_parity_difference
from ignite.exceptions import NotComputableError
from ignite.metrics.fairness.demographic_parity import DemographicParityDifference, SelectionRate


def test_demographic_parity_difference_empty() -> None:
    """Tests if NotComputableError is raised when no data is provided."""
    metric = DemographicParityDifference(groups=[0, 1])
    with pytest.raises(NotComputableError, match="Fairness metrics must have at least one example"):
        metric.compute()


def test_demographic_parity_difference_single_group() -> None:
    """Tests if NotComputableError is raised when only one subgroup is present."""
    metric = DemographicParityDifference(groups=[0])
    y_pred = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    y = torch.tensor([0, 0])
    group_labels = torch.tensor([0, 0])
    metric.update((y_pred, y, group_labels))
    with pytest.raises(NotComputableError, match="Fairness metrics require at least two unique subgroups"):
        metric.compute()


def test_demographic_parity_difference_binary_probs_shape_B(available_device) -> None:
    """Tests DemographicParityDifference with shape (B,) thresholded inputs."""
    metric = DemographicParityDifference(groups=[0, 1], device=available_device)
    # y_pred is (B,) already thresholded
    # Group 0: 1 pos / 2 total = 0.5
    # Group 1: 0 pos / 2 total = 0.0
    y_pred = torch.tensor([1, 0, 0, 0], device=available_device)
    y = torch.tensor([0, 0, 0, 0], device=available_device)  # ignored
    groups = torch.tensor([0, 0, 1, 1], device=available_device)

    metric.update((y_pred, y, groups))
    assert_close(metric.compute(), 0.5)


def test_demographic_parity_difference_binary_probs_shape_B_1(available_device) -> None:
    """Tests DemographicParityDifference with shape (B, 1) thresholded inputs."""
    metric = DemographicParityDifference(groups=[0, 1], device=available_device)
    # y_pred is (B, 1) already thresholded
    y_pred = torch.tensor([[1], [0], [1], [0]], device=available_device)
    y = torch.tensor([0, 0, 0, 0], device=available_device)
    groups = torch.tensor([0, 0, 1, 1], device=available_device)
    metric.update((y_pred, y, groups))
    # G0 selection rate: 0.5, G1 selection rate: 0.5 -> Diff 0.0
    assert_close(metric.compute(), 0.0)


def test_demographic_parity_difference_multiclass(available_device) -> None:
    """Tests DemographicParityDifference with multiclass logits."""
    metric = DemographicParityDifference(groups=[0, 1], device=available_device)
    # y_pred is (B, C)
    # G0 selection rates: [0.5, 0.5, 0.0]
    # G1 selection rates: [0.5, 0.0, 0.5]
    y_pred = torch.tensor(
        [
            [0.8, 0.1, 0.1],  # pred class 0
            [0.1, 0.8, 0.1],  # pred class 1
            [0.8, 0.1, 0.1],  # pred class 0
            [0.1, 0.1, 0.8],  # pred class 2
        ],
        device=available_device,
    )
    y = torch.tensor([0, 0, 0, 0], device=available_device)
    groups = torch.tensor([0, 0, 1, 1], device=available_device)

    metric.update((y_pred, y, groups))
    # Disparities: Class 0: 0.0, Class 1: 0.5, Class 2: 0.5
    assert_close(metric.compute(), 0.5)


def test_demographic_parity_difference_multilabel(available_device) -> None:
    """Tests DemographicParityDifference with multilabel data."""
    metric = DemographicParityDifference(groups=[0, 1], is_multilabel=True, device=available_device)
    # y_pred is (B, C) indicators
    # G0: [1, 1, 0], [0, 0, 0] -> rates: [0.5, 0.5, 0.0]
    # G1: [1, 1, 1], [0, 1, 0] -> rates: [0.5, 1.0, 0.5]
    y_pred = torch.tensor([[1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0]], device=available_device)
    y = torch.tensor([[0, 0, 0]] * 4, device=available_device)
    groups = torch.tensor([0, 0, 1, 1], device=available_device)

    metric.update((y_pred, y, groups))
    # Disparities: C0: 0.0, C1: 0.5, C2: 0.5
    assert_close(metric.compute(), 0.5)


def test_compare_demographic_parity_with_fairlearn(available_device) -> None:
    """Verifies DemographicParityDifference matches Fairlearn's demographic_parity_difference"""
    ignite_metric = DemographicParityDifference(groups=[0, 1], device=available_device)

    # Multi-class case
    # G0: [0, 1, 0], G1: [1, 1, 1]
    y_pred_probs = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
        ],
        device=available_device,
    )
    y_true = torch.zeros(6, device=available_device)  # ignored by SelectionRate
    group_labels = torch.tensor([0, 0, 0, 1, 1, 1], device=available_device)

    ignite_metric.update((y_pred_probs, y_true, group_labels))
    ignite_result = ignite_metric.compute()

    # Fairlearn verification for multiclass
    # We calculate demographic parity difference for each class independently (one-vs-rest)
    # and take the max, which is how Ignite's SubgroupDifference handles vector outputs.
    y_pred_classes = torch.argmax(y_pred_probs, dim=1).cpu().numpy()
    fairlearn_max_diff = 0.0
    for c in range(3):
        y_pred_bin = (y_pred_classes == c).astype(int)
        diff = demographic_parity_difference(
            y_true=y_true.cpu().numpy(), y_pred=y_pred_bin, sensitive_features=group_labels.cpu().numpy()
        )
        fairlearn_max_diff = max(fairlearn_max_diff, diff)

    assert_close(ignite_result, float(fairlearn_max_diff))

    # Simple binary case
    y_pred_binary = torch.tensor([1, 0, 1, 0, 0, 0], device=available_device)
    group_labels_binary = torch.tensor([0, 0, 0, 1, 1, 1], device=available_device)

    ignite_metric.reset()
    ignite_metric.update((y_pred_binary, y_true, group_labels_binary))
    ignite_res_bin = ignite_metric.compute()

    fairlearn_res_bin = demographic_parity_difference(
        y_true=y_true.cpu().numpy(),
        y_pred=y_pred_binary.cpu().numpy(),
        sensitive_features=group_labels_binary.cpu().numpy(),
    )

    assert_close(ignite_res_bin, float(fairlearn_res_bin))


def test_selection_rate_binary(available_device) -> None:
    """Tests SelectionRate for basic binary predictions."""
    metric = SelectionRate(device=available_device)
    # 2 positives out of 4
    y_pred = torch.tensor([1, 0, 1, 0], device=available_device)
    y = torch.tensor([0, 0, 0, 0], device=available_device)
    metric.update((y_pred, y))
    res = metric.compute()
    assert_close(res, torch.tensor([0.5, 0.5], device=available_device))


def test_selection_rate_multiclass(available_device) -> None:
    """Tests SelectionRate for multiclass predictions."""
    metric = SelectionRate(device=available_device)
    # 0: 2 picks, 1: 1 pick, 2: 1 pick. Total 4.
    y_pred = torch.tensor([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], device=available_device)
    y = torch.tensor([0, 0, 0, 0], device=available_device)
    metric.update((y_pred, y))
    res = metric.compute()
    # proportions: [0.5, 0.25, 0.25]
    assert_close(res, torch.tensor([0.5, 0.25, 0.25], device=available_device))


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_compute_demographic_parity_difference(self) -> None:
        rank = idist.get_rank()
        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            metric = DemographicParityDifference(groups=[0, 1], device=metric_device)

            y = torch.tensor([0, 0], device=device)
            groups = torch.tensor([0, 1], device=device)

            if rank == 0:
                # G0: pos, G1: neg
                y_pred = torch.tensor([[0.2, 0.8], [0.8, 0.2]], device=device)
            else:
                # G0: pos, G1: neg
                y_pred = torch.tensor([[0.1, 0.9], [0.9, 0.1]], device=device)

            metric.update((y_pred, y, groups))
            res = metric.compute()

            # Across all ranks:
            # Group 0 selected rate: 1.0
            # Group 1 selected rate: 0.0
            # Max Disparity = 1.0
            assert_close(res, 1.0)


def test_demographic_parity_difference_validation() -> None:
    """Tests input validation for DemographicParityDifference."""
    metric = DemographicParityDifference(groups=[0, 1])

    # Valid multiclass logits
    y_pred = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],  # Group 0: all class 1
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],  # Group 1: all class 0
        ]
    )
    y = torch.tensor([1, 1, 0, 0])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    # Class 0: Group 0 rate = 0, Group 1 rate = 1.0 -> Diff = 1.0
    # Class 1: Group 0 rate = 1.0, Group 1 rate = 0 -> Diff = 1.0
    assert metric.compute() == 1.0

    metric.reset()

    # Valid binary classes (0/1)
    y_pred = torch.tensor([1, 1, 0, 0])
    y = torch.tensor([1, 1, 0, 0])
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    assert metric.compute() == 1.0

    metric.reset()

    # Invalid binary probabilities
    y_pred = torch.tensor([0.6, 0.4, 0.8, 0.1])
    y = torch.tensor([1, 0, 1, 0])
    groups = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="y_pred must be comprised of 0's and 1's"):
        metric.update((y_pred, y, groups))


def test_demographic_parity_multilabel_validation() -> None:
    """Tests input validation for DemographicParityDifference multilabel data."""
    metric = DemographicParityDifference(groups=[0, 1], is_multilabel=True)

    # Valid multilabel (0/1)
    # y_pred is (B, C) indicators
    y_pred = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 1]])
    y = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]])  # Ignored by SelectionRate
    groups = torch.tensor([0, 0, 1, 1])
    metric.update((y_pred, y, groups))
    assert metric.compute() >= 0

    metric.reset()

    # Invalid multilabel (not 0/1)
    y_pred = torch.tensor([[0.6, 0.4], [0.8, 0.1]])
    y = torch.tensor([[1, 0], [1, 1]])
    groups = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="y_pred must be comprised of 0's and 1's"):
        metric.update((y_pred, y, groups))


def test_demographic_parity_shape_mismatch_validation() -> None:
    """Tests validation for shape mismatches between y and y_pred."""
    metric = DemographicParityDifference(groups=[0, 1])

    # y is (B,), y_pred is (B, C) - OK
    metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(4)))

    # y is (B,), y_pred is (B, C, H, W) - Error (dimension mismatch)
    with pytest.raises(ValueError, match="y must have shape of"):
        metric.update((torch.randn(4, 3, 2, 2), torch.randint(0, 3, (4,)), torch.zeros(4)))


def test_demographic_parity_batch_size_mismatch_validation() -> None:
    """Tests validation for batch size mismatches across y_pred, y, and groups."""
    metric = DemographicParityDifference(groups=[0, 1])

    # y_pred (4), y (3) - Batch size mismatch
    with pytest.raises(ValueError, match="y_pred, y, and group_labels must have the same batch size"):
        metric.update((torch.randn(4, 3), torch.randint(0, 3, (3,)), torch.zeros(4)))

    # groups (3), y_pred (4) - Batch size mismatch
    with pytest.raises(ValueError, match="y_pred, y, and group_labels must have the same batch size"):
        metric.update((torch.randn(4, 3), torch.randint(0, 3, (4,)), torch.zeros(3)))


def test_demographic_parity_type_switch_validation() -> None:
    """Tests if RuntimeError is raised when input type changes mid-epoch."""
    metric = DemographicParityDifference(groups=[0, 1])

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


def test_demographic_parity_difference_checkpointing() -> None:
    """Verifies that the state of SubgroupDifference is correctly preserved across state_dict calls."""
    metric = DemographicParityDifference(groups=[0, 1])

    # 1. Update with some data for Group 0
    y_pred_0 = torch.tensor([[0.1, 0.9], [0.1, 0.9]])
    y_true_0 = torch.tensor([1, 1])
    groups_0 = torch.tensor([0, 0])
    metric.update((y_pred_0, y_true_0, groups_0))

    # 2. Save the state
    state = metric.state_dict()

    # 3. Create a NEW metric instance and load the state
    new_metric = DemographicParityDifference(groups=[0, 1])
    new_metric.load_state_dict(state)

    # 4. Update the NEW metric with data for Group 1 only
    y_pred_1 = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    y_true_1 = torch.tensor([1, 1])
    groups_1 = torch.tensor([1, 1])
    new_metric.update((y_pred_1, y_true_1, groups_1))

    # 5. Compute the result on the new metric
    # The new metric should have 1.0 selection rate for Group 0 (from state)
    # and 0.0 selection rate for Group 1 (from new update).
    # Result: 1.0 - 0.0 = 1.0
    assert new_metric.compute() == 1.0
