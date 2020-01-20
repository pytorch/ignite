import os

import torch
import torch.nn as nn

from ignite.engine import Events, Engine
from ignite.contrib.engines.common import setup_common_training_handlers, \
    save_best_model_by_val_score, add_early_stopping_by_val_score, setup_tb_logging

from ignite.handlers import TerminateOnNan
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler

import pytest
from unittest.mock import MagicMock


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


def _test_setup_common_training_handlers(dirname, device, rank=0, local_rank=0, distributed=False):

    lr = 0.01
    step_size = 100
    gamma = 0.5

    model = DummyModel().to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank, ],
                                                          output_device=local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    def update_fn(engine, batch):
        optimizer.zero_grad()
        x = torch.tensor([batch], requires_grad=True, device=device)
        y_pred = model(x)
        loss = y_pred.mean()
        loss.backward()
        optimizer.step()
        return loss

    train_sampler = MagicMock()
    train_sampler.set_epoch = MagicMock()

    trainer = Engine(update_fn)
    setup_common_training_handlers(trainer, train_sampler=train_sampler,
                                   to_save={"model": model, "optimizer": optimizer},
                                   save_every_iters=75, output_path=dirname,
                                   lr_scheduler=lr_scheduler, with_gpu_stats=False,
                                   output_names=['batch_loss', ],
                                   with_pbars=True, with_pbar_on_iters=True, log_every_iters=50,
                                   device=device)

    num_iters = 100
    num_epochs = 10
    data = [i * 0.1 for i in range(num_iters)]
    trainer.run(data, max_epochs=num_epochs)

    # check handlers
    handlers = trainer._event_handlers[Events.ITERATION_COMPLETED]
    for cls in [TerminateOnNan, ]:
        assert any([isinstance(h[0], cls) for h in handlers]), "{}".format(handlers)
    assert 'batch_loss' in trainer.state.metrics

    # Check saved checkpoint
    if rank == 0:
        checkpoints = list(os.listdir(dirname))
        assert len(checkpoints) == 1
        for v in ["training_checkpoint", ]:
            assert any([v in c for c in checkpoints])

    # Check LR scheduling
    assert optimizer.param_groups[0]['lr'] <= lr * gamma ** (num_iters * num_epochs / step_size), \
        "{} vs {}".format(optimizer.param_groups[0]['lr'], lr * gamma ** (num_iters * num_epochs / step_size))


def test_asserts_setup_common_training_handlers():
    trainer = Engine(lambda e, b: None)

    with pytest.raises(ValueError, match=r"If to_save argument is provided then output_path argument should be "
                                         r"also defined"):
        setup_common_training_handlers(trainer, to_save={})

    with pytest.warns(UserWarning, match=r"Argument train_sampler distributed sampler used to call "
                                         r"`set_epoch` method on epoch"):
        train_sampler = MagicMock()
        setup_common_training_handlers(trainer, train_sampler=train_sampler, with_gpu_stats=False)


def test_setup_common_training_handlers(dirname, capsys):

    _test_setup_common_training_handlers(dirname, device='cpu')

    # Check epoch-wise pbar
    captured = capsys.readouterr()
    out = captured.err.split('\r')
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    assert u"Epoch:" in out[-1], "{}".format(out[-1])


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


def test_setup_tb_logging(dirname):

    def _test(with_eval, with_optim):
        trainer = Engine(lambda e, b: b)
        evaluators = None
        optimizers = None

        if with_eval:
            evaluator = Engine(lambda e, b: None)
            acc_scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2, 0.1, 0.1, 0.0]

            @trainer.on(Events.EPOCH_COMPLETED)
            def validate(engine):
                evaluator.run([0, ])

            @evaluator.on(Events.EPOCH_COMPLETED)
            def set_eval_metric(engine):
                engine.state.metrics = {"acc": acc_scores[trainer.state.epoch - 1]}

            evaluators = {'validation': evaluator}

        if with_optim:
            t = torch.tensor([0, ])
            optimizers = {'optimizer': torch.optim.SGD([t, ], lr=0.01)}

        setup_tb_logging(dirname, trainer, optimizers=optimizers, evaluators=evaluators, log_every_iters=1)

        handlers = trainer._event_handlers[Events.ITERATION_COMPLETED]
        for cls in [OutputHandler, ]:
            assert any([isinstance(h[0], cls) for h in handlers]), "{}".format(handlers)

        if with_optim:
            handlers = trainer._event_handlers[Events.ITERATION_STARTED]
            for cls in [OptimizerParamsHandler, ]:
                assert any([isinstance(h[0], cls) for h in handlers]), "{}".format(handlers)

        if with_eval:
            handlers = evaluator._event_handlers[Events.COMPLETED]
            for cls in [OutputHandler, ]:
                assert any([isinstance(h[0], cls) for h in handlers]), "{}".format(handlers)

        data = [0, 1, 2]
        trainer.run(data, max_epochs=10)

        tb_files = list(os.listdir(dirname))
        assert len(tb_files) == 1
        for v in ["events", ]:
            assert any([v in c for c in tb_files]), "{}".format(tb_files)

    _test(with_eval=False, with_optim=False)
    _test(with_eval=True, with_optim=True)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(dirname, distributed_context_single_node_nccl):
    local_rank = distributed_context_single_node_nccl['local_rank']
    device = "cuda:{}".format(local_rank)
    _test_setup_common_training_handlers(dirname, device, rank=local_rank, local_rank=local_rank, distributed=True)
    test_add_early_stopping_by_val_score()


@pytest.mark.distributed
def test_distrib_cpu(dirname, distributed_context_single_node_gloo):
    device = "cpu"
    local_rank = distributed_context_single_node_gloo['local_rank']
    _test_setup_common_training_handlers(dirname, device, rank=local_rank)
    test_add_early_stopping_by_val_score()


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(dirname, distributed_context_multi_node_gloo):
    device = "cpu"
    rank = distributed_context_multi_node_gloo['rank']
    _test_setup_common_training_handlers(dirname, device, rank=rank)
    test_add_early_stopping_by_val_score()


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(dirname, distributed_context_multi_node_nccl):
    local_rank = distributed_context_multi_node_nccl['local_rank']
    rank = distributed_context_multi_node_nccl['rank']
    device = "cuda:{}".format(local_rank)
    _test_setup_common_training_handlers(dirname, device, rank=rank, local_rank=local_rank, distributed=True)
    test_add_early_stopping_by_val_score()
