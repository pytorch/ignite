import warnings
from functools import partial
from itertools import accumulate

import numpy as np
import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, RunningAverage
from ignite.metrics.metric import RunningBatchWise, RunningEpochWise, SingleEpochRunningBatchWise


def test_wrong_input_args():
    with pytest.raises(TypeError, match=r"Argument src should be a Metric or None."):
        RunningAverage(src=[12, 34])

    with pytest.raises(ValueError, match=r"Argument alpha should be a float between"):
        RunningAverage(alpha=-1.0)

    with pytest.raises(ValueError, match=r"Argument output_transform should be None if src is a Metric"):
        RunningAverage(Accuracy(), output_transform=lambda x: x[0])

    with pytest.raises(ValueError, match=r"Argument output_transform should not be None if src corresponds"):
        RunningAverage()

    with pytest.raises(ValueError, match=r"Argument device should be None if src is a Metric"):
        RunningAverage(Accuracy(), device="cpu")

    with pytest.warns(UserWarning, match=r"`epoch_bound` is deprecated and will be removed in the future."):
        m = RunningAverage(Accuracy(), epoch_bound=True)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("epoch_bound, usage", [(False, RunningBatchWise()), (True, SingleEpochRunningBatchWise())])
def test_epoch_bound(epoch_bound, usage):
    with warnings.catch_warnings():
        metric = RunningAverage(output_transform=lambda _: _, epoch_bound=epoch_bound)
    e1 = Engine(lambda _, __: None)
    e2 = Engine(lambda _, __: None)
    metric.attach(e1, "")
    metric.epoch_bound = None
    metric.attach(e2, "", usage)
    e1._event_handlers == e2._event_handlers


@pytest.mark.parametrize("usage", [RunningBatchWise(), SingleEpochRunningBatchWise()])
def test_integration_batchwise(usage):
    torch.manual_seed(10)
    alpha = 0.98
    n_iters = 10
    batch_size = 10
    n_classes = 10
    max_epochs = 3
    data = list(range(n_iters))
    loss = torch.arange(n_iters, dtype=torch.float)
    y_true = torch.randint(0, n_classes, size=(n_iters, batch_size))
    y_pred = torch.rand(n_iters, batch_size, n_classes)

    accuracy_running_averages = torch.tensor(
        list(
            accumulate(
                map(
                    lambda y_yp: torch.sum(y_yp[1].argmax(dim=-1) == y_yp[0]).item() / y_yp[0].size(0),
                    zip(
                        y_true if isinstance(usage, SingleEpochRunningBatchWise) else y_true.repeat(max_epochs, 1),
                        y_pred if isinstance(usage, SingleEpochRunningBatchWise) else y_pred.repeat(max_epochs, 1, 1),
                    ),
                ),
                lambda ra, acc: ra * alpha + (1 - alpha) * acc,
            )
        )
    )
    if isinstance(usage, SingleEpochRunningBatchWise):
        accuracy_running_averages = accuracy_running_averages.repeat(max_epochs)

    loss_running_averages = torch.tensor(
        list(
            accumulate(
                loss if isinstance(usage, SingleEpochRunningBatchWise) else loss.repeat(max_epochs),
                lambda ra, loss_item: ra * alpha + (1 - alpha) * loss_item,
            )
        )
    )
    if isinstance(usage, SingleEpochRunningBatchWise):
        loss_running_averages = loss_running_averages.repeat(max_epochs)

    def update_fn(_, i):
        loss_value = loss[i]
        y_true_batch = y_true[i]
        y_pred_batch = y_pred[i]
        return loss_value, y_pred_batch, y_true_batch

    trainer = Engine(update_fn)

    acc_metric = RunningAverage(Accuracy(output_transform=lambda x: [x[1], x[2]]), alpha=alpha)
    acc_metric.attach(trainer, "running_avg_accuracy", usage)

    avg_output = RunningAverage(output_transform=lambda x: x[0], alpha=alpha)
    avg_output.attach(trainer, "running_avg_loss", usage)

    metric_acc_running_averages = []
    metric_loss_running_averages = []

    @trainer.on(Events.ITERATION_COMPLETED)
    def _(engine):
        metric_acc_running_averages.append(engine.state.metrics["running_avg_accuracy"])
        metric_loss_running_averages.append(engine.state.metrics["running_avg_loss"])

    trainer.run(data, max_epochs=3)

    assert (torch.tensor(metric_acc_running_averages) == accuracy_running_averages).all()
    assert (torch.tensor(metric_loss_running_averages) == loss_running_averages).all()

    metric_state = acc_metric.state_dict()
    saved__value = acc_metric._value
    saved_src__num_correct = acc_metric.src._num_correct
    saved_src__num_examples = acc_metric.src._num_examples
    acc_metric.reset()
    acc_metric.load_state_dict(metric_state)
    assert acc_metric._value == saved__value
    assert acc_metric.src._num_examples == saved_src__num_examples
    assert (acc_metric.src._num_correct == saved_src__num_correct).all()

    metric_state = avg_output.state_dict()
    saved__value = avg_output._value
    assert avg_output.src is None
    avg_output.reset()
    avg_output.load_state_dict(metric_state)
    assert avg_output._value == saved__value
    assert avg_output.src is None


def test_integration_epochwise():
    torch.manual_seed(10)
    alpha = 0.98
    n_iters = 10
    batch_size = 10
    n_classes = 10
    max_epochs = 3
    data = list(range(n_iters))
    y_true = torch.randint(0, n_classes, size=(n_iters, batch_size))
    y_pred = torch.rand(max_epochs, n_iters, batch_size, n_classes)

    accuracy_running_averages = torch.tensor(
        list(
            accumulate(
                map(
                    lambda y_pred_epoch: torch.sum(y_pred_epoch.argmax(dim=-1) == y_true).item() / y_true.numel(),
                    y_pred,
                ),
                lambda ra, acc: ra * alpha + (1 - alpha) * acc,
            )
        )
    )

    def update_fn(engine, i):
        y_true_batch = y_true[i]
        y_pred_batch = y_pred[engine.state.epoch - 1, i]
        return y_pred_batch, y_true_batch

    trainer = Engine(update_fn)

    acc_metric = RunningAverage(Accuracy(), alpha=alpha)
    acc_metric.attach(trainer, "running_avg_accuracy", RunningEpochWise())

    metric_acc_running_averages = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def _(engine):
        metric_acc_running_averages.append(engine.state.metrics["running_avg_accuracy"])

    trainer.run(data, max_epochs=3)

    assert (torch.tensor(metric_acc_running_averages) == accuracy_running_averages).all()


@pytest.mark.parametrize("usage", [RunningBatchWise(), SingleEpochRunningBatchWise(), RunningEpochWise()])
def test_multiple_attach(usage):
    n_iters = 100
    errD_values = iter(np.random.rand(n_iters))
    errG_values = iter(np.random.rand(n_iters))
    D_x_values = iter(np.random.rand(n_iters))
    D_G_z1 = iter(np.random.rand(n_iters))
    D_G_z2 = iter(np.random.rand(n_iters))

    def update_fn(engine, batch):
        return {
            "errD": next(errD_values),
            "errG": next(errG_values),
            "D_x": next(D_x_values),
            "D_G_z1": next(D_G_z1),
            "D_G_z2": next(D_G_z2),
        }

    trainer = Engine(update_fn)
    alpha = 0.98

    # attach running average
    monitoring_metrics = ["errD", "errG", "D_x", "D_G_z1", "D_G_z2"]
    for metric in monitoring_metrics:
        foo = partial(lambda x, metric: x[metric], metric=metric)
        RunningAverage(alpha=alpha, output_transform=foo).attach(trainer, metric, usage)

    @trainer.on(usage.COMPLETED)
    def check_values(engine):
        values = []
        for metric in monitoring_metrics:
            values.append(engine.state.metrics[metric])

        values = set(values)
        assert len(values) == len(monitoring_metrics)

    data = list(range(n_iters))
    trainer.run(data)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("epoch_bound", [True, False, None])
@pytest.mark.parametrize("src", [Accuracy(), None])
@pytest.mark.parametrize("usage", [RunningBatchWise(), SingleEpochRunningBatchWise(), RunningEpochWise()])
def test_detach(epoch_bound, src, usage):
    with warnings.catch_warnings():
        m = RunningAverage(src, output_transform=(lambda _: _) if src is None else None, epoch_bound=epoch_bound)
    e = Engine(lambda _, __: None)
    m.attach(e, "m", usage)
    for event_handlers in e._event_handlers.values():
        assert len(event_handlers) != 0
    m.detach(e, usage)
    for event_handlers in e._event_handlers.values():
        assert len(event_handlers) == 0


def test_output_is_tensor():
    m = RunningAverage(output_transform=lambda x: x)
    m.update(torch.rand(10, requires_grad=True).mean())
    v = m.compute()
    assert isinstance(v, torch.Tensor)
    assert not v.requires_grad

    m.update(torch.rand(10, requires_grad=True).mean())
    v = m.compute()
    assert isinstance(v, torch.Tensor)
    assert not v.requires_grad

    m.update(torch.rand(10, requires_grad=True).mean())
    v = m.compute()
    assert isinstance(v, torch.Tensor)
    assert not v.requires_grad


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    @pytest.mark.parametrize("usage", [RunningBatchWise(), SingleEpochRunningBatchWise()])
    def test_src_is_output(self, usage):
        device = idist.device()
        rank = idist.get_rank()
        n_iters = 10
        n_epochs = 3

        # Data per rank
        data = list(range(n_iters))
        rank_loss_count = n_epochs * n_iters
        all_loss_values = torch.arange(0, rank_loss_count * idist.get_world_size(), dtype=torch.float64).to(device)
        loss_values = iter(all_loss_values[rank_loss_count * rank : rank_loss_count * (rank + 1)])

        def update_fn(engine, batch):
            loss_value = next(loss_values)
            return loss_value.item()

        trainer = Engine(update_fn)
        alpha = 0.98

        metric_device = device if device.type != "xla" else "cpu"
        avg_output = RunningAverage(output_transform=lambda x: x, alpha=alpha, device=metric_device)
        avg_output.attach(trainer, "running_avg_output", usage)

        @trainer.on(usage.STARTED)
        def reset_running_avg_output(engine):
            engine.state.running_avg_output = None

        @trainer.on(usage.ITERATION_COMPLETED)
        def running_avg_output_update(engine):
            i = engine.state.iteration - 1
            o = sum([all_loss_values[i + r * rank_loss_count] for r in range(idist.get_world_size())]).item()
            o /= idist.get_world_size()
            if engine.state.running_avg_output is None:
                engine.state.running_avg_output = o
            else:
                engine.state.running_avg_output = engine.state.running_avg_output * alpha + (1.0 - alpha) * o

        @trainer.on(usage.COMPLETED)
        def assert_equal_running_avg_output_values(engine):
            it = engine.state.iteration
            assert (
                engine.state.running_avg_output == engine.state.metrics["running_avg_output"]
            ), f"{it}: {engine.state.running_avg_output} vs {engine.state.metrics['running_avg_output']}"

        trainer.run(data, max_epochs=3)

    @pytest.mark.parametrize("usage", [RunningBatchWise(), SingleEpochRunningBatchWise(), RunningEpochWise()])
    def test_src_is_metric(self, usage):
        device = idist.device()
        rank = idist.get_rank()
        n_iters = 10
        n_epochs = 3
        batch_size = 10
        n_classes = 10

        def _test(metric_device):
            data = list(range(n_iters))
            np.random.seed(12)
            all_y_true_batch_values = np.random.randint(
                0, n_classes, size=(idist.get_world_size(), n_epochs * n_iters, batch_size)
            )
            all_y_pred_batch_values = np.random.rand(idist.get_world_size(), n_epochs * n_iters, batch_size, n_classes)

            y_true_batch_values = iter(all_y_true_batch_values[rank, ...])
            y_pred_batch_values = iter(all_y_pred_batch_values[rank, ...])

            def update_fn(engine, batch):
                y_true_batch = next(y_true_batch_values)
                y_pred_batch = next(y_pred_batch_values)
                return torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

            trainer = Engine(update_fn)
            alpha = 0.98

            acc_metric = RunningAverage(Accuracy(device=metric_device), alpha=alpha)
            acc_metric.attach(trainer, "running_avg_accuracy", usage)

            running_avg_acc = [
                None,
            ]
            true_acc_metric = Accuracy(device=metric_device)

            @trainer.on(Events.ITERATION_COMPLETED)
            def manual_running_avg_acc(engine):
                iteration = engine.state.iteration

                if not isinstance(usage, RunningEpochWise) or ((iteration - 1) % n_iters) == 0:
                    true_acc_metric.reset()
                if ((iteration - 1) % n_iters) == 0 and isinstance(usage, SingleEpochRunningBatchWise):
                    running_avg_acc[0] = None
                for j in range(idist.get_world_size()):
                    output = (
                        torch.from_numpy(all_y_pred_batch_values[j, iteration - 1, :, :]),
                        torch.from_numpy(all_y_true_batch_values[j, iteration - 1, :]),
                    )
                    true_acc_metric.update(output)

                if not isinstance(usage, RunningEpochWise) or (iteration % n_iters) == 0:
                    batch_acc = true_acc_metric._num_correct.item() * 1.0 / true_acc_metric._num_examples

                    if running_avg_acc[0] is None:
                        running_avg_acc[0] = batch_acc
                    else:
                        running_avg_acc[0] = running_avg_acc[0] * alpha + (1.0 - alpha) * batch_acc
                    engine.state.running_avg_acc = running_avg_acc[0]

            @trainer.on(Events.ITERATION_COMPLETED)
            def assert_equal_running_avg_acc_values(engine):
                print(engine.state.iteration)
                if not isinstance(usage, RunningEpochWise) or (
                    (engine.state.iteration > 1) and ((engine.state.iteration % n_iters) == 1)
                ):
                    assert (
                        engine.state.running_avg_acc == engine.state.metrics["running_avg_accuracy"]
                    ), f"{engine.state.running_avg_acc} vs {engine.state.metrics['running_avg_accuracy']}"

            trainer.run(data, max_epochs=3)

        _test("cpu")
        if device.type != "xla":
            _test(idist.device())

    def test_accumulator_device(self):
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(idist.device())
        for metric_device in metric_devices:
            # Don't test the src=Metric case because compute() returns a scalar,
            # so the metric doesn't accumulate on the device specified
            avg = RunningAverage(output_transform=lambda x: x, device=metric_device)
            assert avg._device == metric_device
            # Value is None until the first update then compute call

            for _ in range(3):
                avg.update(torch.tensor(1.0, device=device))
                avg.compute()

                assert (
                    avg._value.device == metric_device
                ), f"{type(avg._value.device)}:{avg._value.device} vs {type(metric_device)}:{metric_device}"
