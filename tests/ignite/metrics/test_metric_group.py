import pytest
import torch

from ignite import distributed as idist
from ignite.engine import Engine
from ignite.metrics import Accuracy, MetricGroup, Precision

torch.manual_seed(41)


def test_update():
    precision = Precision()
    accuracy = Accuracy()

    group = MetricGroup({"precision": Precision(), "accuracy": Accuracy()})

    y_pred = torch.randint(0, 2, (100,))
    y = torch.randint(0, 2, (100,))

    precision.update((y_pred, y))
    accuracy.update((y_pred, y))
    group.update((y_pred, y))

    assert precision.state_dict() == group.metrics["precision"].state_dict()
    assert accuracy.state_dict() == group.metrics["accuracy"].state_dict()


def test_output_transform():
    def drop_first(output):
        y_pred, y = output
        return (y_pred[1:], y[1:])

    precision = Precision(output_transform=drop_first)
    accuracy = Accuracy(output_transform=drop_first)

    group = MetricGroup(
        {"precision": Precision(output_transform=drop_first), "accuracy": Accuracy(output_transform=drop_first)}
    )

    y_pred = torch.randint(0, 2, (100,))
    y = torch.randint(0, 2, (100,))

    precision.update(drop_first(drop_first((y_pred, y))))
    accuracy.update(drop_first(drop_first((y_pred, y))))
    group.update(drop_first((y_pred, y)))

    assert precision.state_dict() == group.metrics["precision"].state_dict()
    assert accuracy.state_dict() == group.metrics["accuracy"].state_dict()


def test_compute():
    precision = Precision()
    accuracy = Accuracy()

    group = MetricGroup({"precision": Precision(), "accuracy": Accuracy()})

    for _ in range(3):
        y_pred = torch.randint(0, 2, (100,))
        y = torch.randint(0, 2, (100,))

        precision.update((y_pred, y))
        accuracy.update((y_pred, y))
        group.update((y_pred, y))

    assert group.compute() == {"precision": precision.compute(), "accuracy": accuracy.compute()}

    precision.reset()
    accuracy.reset()
    group.reset()

    assert precision.state_dict() == group.metrics["precision"].state_dict()
    assert accuracy.state_dict() == group.metrics["accuracy"].state_dict()


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_integration(self):
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        n_epochs = 3
        n_iters = 5
        batch_size = 10
        device = idist.device()

        y_true = torch.randint(0, 2, size=(n_iters * batch_size,)).to(device)
        y_pred = torch.randint(0, 2, (n_iters * batch_size,)).to(device)

        def update(_, i):
            return (
                y_pred[i * batch_size : (i + 1) * batch_size],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        precision = Precision()
        precision.attach(engine, "precision")

        accuracy = Accuracy()
        accuracy.attach(engine, "accuracy")

        group = MetricGroup({"eval_metrics.accuracy": Accuracy(), "eval_metrics.precision": Precision()})
        group.attach(engine, "eval_metrics")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "eval_metrics" in engine.state.metrics
        assert "eval_metrics.accuracy" in engine.state.metrics
        assert "eval_metrics.precision" in engine.state.metrics

        assert engine.state.metrics["eval_metrics"] == {
            "eval_metrics.accuracy": engine.state.metrics["accuracy"],
            "eval_metrics.precision": engine.state.metrics["precision"],
        }
        assert engine.state.metrics["eval_metrics.accuracy"] == engine.state.metrics["accuracy"]
        assert engine.state.metrics["eval_metrics.precision"] == engine.state.metrics["precision"]
