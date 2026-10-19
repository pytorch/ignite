from collections import OrderedDict

import pytest
import torch
from torch.testing import assert_close

import ignite.distributed as idist
from ignite.metrics import MetricGroup
from ignite.metrics.fairness import DemographicParityDifference, SubgroupAccuracyDifference


@pytest.mark.parametrize("metric_type", [DemographicParityDifference, SubgroupAccuracyDifference])
@pytest.mark.parametrize("nested", [False, True])
def test_subgroup_checkpointing(metric_type, nested, available_device, tmp_path):
    def make_metric():
        metric = metric_type(groups=[0, 1], device=available_device)
        return MetricGroup({"fairness": metric}) if nested else metric

    metric = make_metric()
    metric.update((torch.tensor([1, 1, 0, 0]), torch.ones(4), torch.tensor([0, 0, 1, 1])))
    checkpoint = tmp_path / "fairness.pt"
    torch.save(metric.state_dict(), checkpoint)

    restored = make_metric()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    expected = {"fairness": 1.0} if nested else 1.0
    assert_close(restored.compute(), expected)

    output = (torch.tensor([0, 1]), torch.ones(2), torch.tensor([0, 1]))
    metric.update(output)
    restored.update(output)
    expected = {"fairness": 1.0 / 3.0} if nested else 1.0 / 3.0
    assert_close(restored.compute(), expected)
    assert_close(restored.compute(), metric.compute())


@pytest.mark.parametrize("metric_type", [DemographicParityDifference, SubgroupAccuracyDifference])
@pytest.mark.parametrize("groups", [[1, 0], [0, 2]])
def test_subgroup_legacy_checkpointing(metric_type, groups):
    metric = metric_type(groups=[0, 1])
    metric.update((torch.tensor([1, 1, 0, 0]), torch.ones(4), torch.tensor([0, 0, 1, 1])))
    state = OrderedDict(
        [
            ("__metric_state_per_rank", [OrderedDict()]),
            ("_metrics", {str(group): child.state_dict() for group, child in metric._metrics.items()}),
        ]
    )

    restored = metric_type(groups=groups)
    if 2 in groups:
        restored.update((torch.tensor([0]), torch.ones(1), torch.tensor([2])))
    restored.load_state_dict(state)
    assert_close(restored.compute(), 1.0)
    assert state["__metric_state_per_rank"] == [OrderedDict()]

    other_group = 2 if 2 in groups else 1
    output = (torch.tensor([0, 1]), torch.ones(2), torch.tensor([0, other_group]))
    restored.update(output)
    assert_close(restored.compute(), 1.0 / 6.0 if 2 in groups else 1.0 / 3.0)


def test_subgroup_legacy_checkpoint_world_size_mismatch():
    metric = SubgroupAccuracyDifference(groups=[0, 1])
    state = {
        "__metric_state_per_rank": [OrderedDict()],
        "_metrics": {"0": {"__metric_state_per_rank": [OrderedDict(), OrderedDict()]}},
    }
    with pytest.raises(ValueError, match="same world size"):
        metric.load_state_dict(state)


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    @pytest.mark.parametrize("metric_type", [DemographicParityDifference, SubgroupAccuracyDifference])
    def test_subgroup_checkpointing(self, metric_type, tmp_path):
        rank = idist.get_rank()
        positives = torch.ones(rank + 1)
        negatives = torch.zeros(rank + 2)
        y_pred = torch.cat([positives, negatives])
        output = (y_pred, torch.ones_like(y_pred), torch.cat([torch.zeros_like(positives), torch.ones_like(negatives)]))

        metric = MetricGroup({"fairness": metric_type(groups=[0, 1])})
        metric.update(output)
        checkpoint = tmp_path / "fairness.pt"
        torch.save(metric.state_dict(), checkpoint)

        restored = MetricGroup({"fairness": metric_type(groups=[0, 1])})
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        children = restored.metrics["fairness"]._metrics
        assert children[0]._num_examples == rank + 1
        assert children[1]._num_examples == rank + 2
        assert_close(restored.compute(), {"fairness": 1.0})

        output = (torch.tensor([0, 1]), torch.ones(2), torch.tensor([0, 1]))
        metric.update(output)
        restored.update(output)
        assert_close(restored.compute(), metric.compute())
