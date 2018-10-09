
import numpy as np
import torch

from ignite.engine import Engine, Events
from ignite.metrics import CategoricalAccuracy, RunningAverage

import pytest


def test_wrong_input_args():
    with pytest.raises(TypeError):
        _ = RunningAverage(src=[12, 34])

    with pytest.raises(ValueError):
        _ = RunningAverage(alpha=-1.0)

    with pytest.raises(ValueError):
        _ = RunningAverage(CategoricalAccuracy(), output_transform=lambda x: x[0])

    with pytest.raises(ValueError):
        _ = RunningAverage()


def test_integration():

    n_iters = 100
    batch_size = 10
    n_classes = 10
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_iters, batch_size, n_classes))
    loss_values = iter(range(n_iters))

    def update_fn(engine, batch):
        loss_value = next(loss_values)
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        return loss_value, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    trainer = Engine(update_fn)
    alpha = 0.98

    acc_metric = RunningAverage(CategoricalAccuracy(output_transform=lambda x: [x[1], x[2]]), alpha=alpha)
    acc_metric.attach(trainer, 'running_avg_accuracy')

    avg_output = RunningAverage(output_transform=lambda x: x[0], alpha=alpha)
    avg_output.attach(trainer, 'running_avg_output')

    running_avg_acc = [None]

    @trainer.on(Events.ITERATION_COMPLETED, running_avg_acc)
    def manual_running_avg_acc(engine, running_avg_acc):
        _, y_pred, y = engine.state.output
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y).view(-1)
        num_correct = torch.sum(correct).item()
        num_examples = correct.shape[0]
        batch_acc = num_correct * 1.0 / num_examples
        if running_avg_acc[0] is None:
            running_avg_acc[0] = batch_acc
        else:
            running_avg_acc[0] = running_avg_acc[0] * alpha + (1.0 - alpha) * batch_acc
        engine.state.running_avg_acc = running_avg_acc[0]

    @trainer.on(Events.EPOCH_STARTED)
    def running_avg_output_init(engine):
        engine.state.running_avg_output = None

    @trainer.on(Events.ITERATION_COMPLETED)
    def running_avg_output_update(engine):
        if engine.state.running_avg_output is None:
            engine.state.running_avg_output = engine.state.output[0]
        else:
            engine.state.running_avg_output = engine.state.running_avg_output * alpha + \
                (1.0 - alpha) * engine.state.output[0]

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_acc_values(engine):
        assert engine.state.running_avg_acc == engine.state.metrics['running_avg_accuracy'], \
            "{} vs {}".format(engine.state.running_avg_acc, engine.state.metrics['running_avg_accuracy'])

    @trainer.on(Events.ITERATION_COMPLETED)
    def assert_equal_running_avg_output_values(engine):
        assert engine.state.running_avg_output == engine.state.metrics['running_avg_output'], \
            "{} vs {}".format(engine.state.running_avg_output, engine.state.metrics['running_avg_output'])

    np.random.seed(10)
    running_avg_acc[0] = None
    n_iters = 10
    batch_size = 10
    n_classes = 10
    data = list(range(n_iters))
    loss_values = iter(range(n_iters))
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_iters, batch_size, n_classes))
    trainer.run(data, max_epochs=1)

    running_avg_acc[0] = None
    n_iters = 10
    batch_size = 10
    n_classes = 10
    data = list(range(n_iters))
    loss_values = iter(range(n_iters))
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_iters, batch_size, n_classes))
    trainer.run(data, max_epochs=1)
