import os
import tempfile
import shutil

import torch
import torch.nn as nn

from ignite.engine import Events
from ignite.contrib.engines import create_common_trainer, create_common_distrib_trainer

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


def test_create_common_trainer(dirname, capsys):

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

    trainer = create_common_trainer(update_fn,
                                    to_save={"model": model, "optimizer": optimizer},
                                    save_every=75, output_path=dirname,
                                    lr_scheduler=lr_scheduler, with_gpu_stats=False,
                                    output_names=['batch_loss', ],
                                    with_pbars=True, with_pbar_on_iters=True, log_every=50)

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
    assert len(checkpoints) == 2
    for v in ["checkpoint_model", "checkpoint_optimizer"]:
        assert any([v in c for c in checkpoints])

    # Check LR scheduling
    assert optimizer.param_groups[0]['lr'] <= lr * gamma ** (num_iters * num_epochs / step_size), \
        "{} vs {}".format(optimizer.param_groups[0]['lr'], lr * gamma ** (num_iters * num_epochs / step_size))
