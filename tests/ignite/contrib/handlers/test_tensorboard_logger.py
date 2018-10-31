import pytest
import torch
import torch.nn as nn
import torch.utils.data as utils

from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import BinaryAccuracy, RunningAverage
from torch.optim import SGD


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def test_logger_fail():
    x = torch.rand(4, 1)
    y = torch.zeros(4).type(torch.LongTensor)
    dset = utils.TensorDataset(x, y)
    data_loader = utils.DataLoader(dset)

    model = Model()
    optimizer = SGD(model.parameters(), lr=1e-1)
    loss_fn = nn.CrossEntropyLoss()

    def step(engine, batch):
        optimizer.zero_grad()
        x, y = batch
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y

    trainer = Engine(step)
    evaluator = create_supervised_evaluator(model=model, metrics={'acc': BinaryAccuracy()})

    accuracy = RunningAverage(BinaryAccuracy(output_transform=lambda x: (x[1], x[2])))
    accuracy.attach(trainer, "avg_accuracy")

    tbLogger = TensorboardLogger()

    with pytest.raises(ValueError):
        tbLogger.attach(engine=trainer,
                        mode='fail',
                        model=model,
                        use_metrics=True,
                        histogram_freq=1,
                        write_grads=True)

    with pytest.raises(TypeError):
        tbLogger.attach(engine=trainer,
                        mode='iteration',
                        model=5,
                        use_metrics=True,
                        histogram_freq=1,
                        write_grads=True)

    with pytest.raises(TypeError):
        tbLogger.attach(engine=trainer,
                        mode='epoch',
                        model=model,
                        use_metrics=None,
                        histogram_freq=1,
                        write_grads=True)

    with pytest.raises(TypeError):
        tbLogger.attach(engine=trainer,
                        mode='epoch',
                        model=model,
                        use_metrics=True,
                        histogram_freq=1,
                        write_grads=6)

    with pytest.raises(TypeError):
        tbLogger.attach(engine=trainer,
                        mode='epoch',
                        model=model,
                        use_metrics=True,
                        histogram_freq=0.5,
                        write_grads=True)


def test_tblogger_state_key():
    x = torch.rand(4, 1)
    y = torch.zeros(4).type(torch.LongTensor)
    dset = utils.TensorDataset(x, y)
    data_loader = utils.DataLoader(dset)

    model = Model()
    optimizer = SGD(model.parameters(), lr=1e-1)
    loss_fn = nn.CrossEntropyLoss()

    def step(engine, batch):
        optimizer.zero_grad()
        x, y = batch
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y

    trainer = Engine(step)
    evaluator = create_supervised_evaluator(model=model, metrics={'acc': BinaryAccuracy()})

    accuracy = RunningAverage(BinaryAccuracy(output_transform=lambda x: (x[1], x[2])))
    accuracy.attach(trainer, "avg_accuracy")

    tbLogger = TensorboardLogger(log_dir='/tmp/')
    tbLogger.attach(engine=trainer,
                    mode='iteration',
                    model=model,
                    use_metrics=True,
                    state_keys=['apple'],
                    histogram_freq=1,
                    write_grads=True)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        tbLogger.write_graph(model, data_loader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validation(engine):
        evaluator.run(data=data_loader, max_epochs=1)
        tbLogger.plot_metrics(evaluator, name='validation', mode='epoch')

    with pytest.raises(KeyError):
        trainer.run(data_loader, max_epochs=1)

    @trainer.on(Events.STARTED)
    def intialize_state(engine):
        engine.state.apple = 'a'

    with pytest.raises(ValueError):
        trainer.run(data_loader, max_epochs=1)


def test_no_error_epoch():
    x = torch.rand(4, 1)
    y = torch.zeros(4).type(torch.LongTensor)
    dset = utils.TensorDataset(x, y)
    data_loader = utils.DataLoader(dset)

    model = Model()
    optimizer = SGD(model.parameters(), lr=1e-1)
    loss_fn = nn.CrossEntropyLoss()

    def step(engine, batch):
        optimizer.zero_grad()
        x, y = batch
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y

    trainer = Engine(step)
    evaluator = create_supervised_evaluator(model=model, metrics={'acc': BinaryAccuracy()})

    accuracy = RunningAverage(BinaryAccuracy(output_transform=lambda x: (x[1], x[2])))
    accuracy.attach(trainer, "avg_accuracy")

    tbLogger = TensorboardLogger(log_dir='/tmp/')
    tbLogger.attach(engine=trainer,
                    mode='epoch',
                    model=model,
                    use_metrics=True,
                    state_keys=['running_reward'],
                    histogram_freq=1,
                    write_grads=True)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        tbLogger.write_graph(model, data_loader)
        engine.state.running_reward = 10

    @trainer.on(Events.EPOCH_COMPLETED)
    def validation(engine):
        evaluator.run(data=data_loader, max_epochs=1)
        tbLogger.plot_metrics(evaluator, name='validation', mode='epoch')

    trainer.run(data_loader, max_epochs=2)


def test_no_error_iteration():
    x = torch.rand(4, 1)
    y = torch.zeros(4).type(torch.LongTensor)
    dset = utils.TensorDataset(x, y)
    data_loader = utils.DataLoader(dset)

    model = Model()
    optimizer = SGD(model.parameters(), lr=1e-1)
    loss_fn = nn.CrossEntropyLoss()

    def step(engine, batch):
        optimizer.zero_grad()
        x, y = batch
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y

    trainer = Engine(step)
    evaluator = create_supervised_evaluator(model=model, metrics={'acc': BinaryAccuracy()})

    accuracy = RunningAverage(BinaryAccuracy(output_transform=lambda x: (x[1], x[2])))
    accuracy.attach(trainer, "avg_accuracy")

    tbLogger = TensorboardLogger(log_dir='/tmp/')
    tbLogger.attach(engine=trainer,
                    mode='iteration',
                    model=model,
                    use_metrics=True,
                    state_keys=['running_reward'],
                    histogram_freq=1,
                    write_grads=True)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        tbLogger.write_graph(model, data_loader)
        engine.state.running_reward = 10

    @trainer.on(Events.EPOCH_COMPLETED)
    def validation(engine):
        evaluator.run(data=data_loader, max_epochs=1)
        tbLogger.plot_metrics(evaluator, name='validation', mode='epoch')

    trainer.run(data_loader, max_epochs=2)
