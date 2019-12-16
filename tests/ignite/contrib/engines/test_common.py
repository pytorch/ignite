import os
import tempfile
import shutil

import torch
import torch.nn as nn

from ignite.engine import Events, Engine
from ignite.contrib.engines.common import setup_common_training_handlers, \
    save_best_model_by_val_score, add_early_stopping_by_val_score

from ignite.handlers import TerminateOnNan

import pytest


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


def test_setup_common_training_handlers(dirname, capsys):

    lr = 0.01
    step_size = 100
    gamma = 0.5

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def update_fn(engine, batch):
        optimizer.zero_grad()
        x = torch.tensor([batch], requires_grad=True)
        y_pred = model(x)
        loss = y_pred.mean()
        loss.backward()
        optimizer.step()
        return loss

    trainer = Engine(update_fn)
    setup_common_training_handlers(trainer,
                                   to_save={"model": model, "optimizer": optimizer},
                                   save_every_iters=75, output_path=dirname,
                                   lr_scheduler=lr_scheduler, with_gpu_stats=False,
                                   output_names=['batch_loss', ],
                                   with_pbars=True, with_pbar_on_iters=True, log_every_iters=50)

    num_iters = 100
    num_epochs = 10
    data = [i * 0.1 for i in range(num_iters)]
    trainer.run(data, max_epochs=num_epochs)

    # check handlers
    handlers = trainer._event_handlers[Events.ITERATION_COMPLETED]
    for cls in [TerminateOnNan, ]:
        assert any([isinstance(h[0], cls) for h in handlers]), \
            "{}".format(trainer._event_handlers[Events.ITERATION_COMPLETED])
    assert 'batch_loss' in trainer.state.metrics

    # Check epoch-wise pbar
    captured = capsys.readouterr()
    out = captured.err.split('\r')
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    assert u"Epoch:" in out[-1], "{}".format(out[-1])

    # Check saved checkpoint
    checkpoints = list(os.listdir(dirname))
    assert len(checkpoints) == 1
    for v in ["training_checkpoint", ]:
        assert any([v in c for c in checkpoints])

    # Check LR scheduling
    assert optimizer.param_groups[0]['lr'] <= lr * gamma ** (num_iters * num_epochs / step_size), \
        "{} vs {}".format(optimizer.param_groups[0]['lr'], lr * gamma ** (num_iters * num_epochs / step_size))


def test_save_best_model_by_val_score(dirname, capsys):

    trainer = Engine(lambda e, b: None)
    evaluator = Engine(lambda e, b: None)
    model = DummyModel()

    acc_scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.5, 0.6, 0.61, 0.7, 0.5]

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        evaluator.run([0, ])

    @evaluator.on(Events.EPOCH_COMPLETED)
    def set_eval_metric(engine):
        engine.state.metrics = {"acc": acc_scores[trainer.state.epoch - 1]}

    save_best_model_by_val_score(dirname, evaluator, model, metric_name="acc", n_saved=2, trainer=trainer)

    data = [0, ]
    trainer.run(data, max_epochs=len(acc_scores))

    assert set(os.listdir(dirname)) == set(['best_model_8_val_acc=0.61.pth', 'best_model_9_val_acc=0.7.pth'])


def test_add_early_stopping_by_val_score():
    trainer = Engine(lambda e, b: None)
    evaluator = Engine(lambda e, b: None)

    acc_scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2, 0.1, 0.1, 0.0]

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        evaluator.run([0, ])

    @evaluator.on(Events.EPOCH_COMPLETED)
    def set_eval_metric(engine):
        engine.state.metrics = {"acc": acc_scores[trainer.state.epoch - 1]}

    add_early_stopping_by_val_score(patience=3, evaluator=evaluator, trainer=trainer, metric_name="acc")

    data = [0, ]
    state = trainer.run(data, max_epochs=len(acc_scores))

    assert state.epoch == 7
