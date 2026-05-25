import os
from unittest.mock import call, MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

import ignite.distributed as idist
from ignite.contrib.engines.common import (
    add_early_stopping_by_val_score,
    gen_save_best_models_by_val_score,
    save_best_model_by_val_score,
    setup_common_training_handlers,
)
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver, TerminateOnNan


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)


def _test_setup_common_training_handlers(
    dirname,
    device,
    rank=0,
    local_rank=0,
    distributed=False,
    lr_scheduler=None,
    save_handler=None,
    output_transform=lambda loss: loss,
):
    lr = 0.01
    step_size = 100
    gamma = 0.5
    num_iters = 100
    num_epochs = 10

    model = DummyModel().to(device)
    if distributed and "cuda" in torch.device(device).type:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif isinstance(lr_scheduler, str) and lr_scheduler == "ignite|LRScheduler":
        from ignite.handlers import LRScheduler

        lr_scheduler = LRScheduler(torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma))
    elif isinstance(lr_scheduler, str) and lr_scheduler == "ignite":
        from ignite.handlers import PiecewiseLinear

        milestones_values = [(0, 0.0), (step_size, lr), (num_iters * (num_epochs - 1), 0.0)]
        lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    def update_fn(engine, batch):
        optimizer.zero_grad()
        x = torch.tensor([batch], requires_grad=True, device=device)
        y_pred = model(x)
        loss = y_pred.mean()
        loss.backward()
        optimizer.step()
        return output_transform(loss)

    train_sampler = None
    if distributed and idist.get_world_size() > 1:
        train_sampler = MagicMock(spec=DistributedSampler)
        train_sampler.set_epoch = MagicMock()

    trainer = Engine(update_fn)
    setup_common_training_handlers(
        trainer,
        train_sampler=train_sampler,
        to_save={"model": model, "optimizer": optimizer},
        save_every_iters=75,
        output_path=dirname,
        save_handler=save_handler,
        lr_scheduler=lr_scheduler,
        with_gpu_stats=False,
        output_names=["batch_loss"],
        with_pbars=True,
        with_pbar_on_iters=True,
        log_every_iters=50,
    )

    data = [i * 0.1 for i in range(num_iters)]
    trainer.run(data, max_epochs=num_epochs)

    # check handlers
    handlers = trainer._event_handlers[Events.ITERATION_COMPLETED]
    for cls in [
        TerminateOnNan,
    ]:
        assert any([isinstance(h[0], cls) for h in handlers]), f"{handlers}"
    assert "batch_loss" in trainer.state.metrics

    # Check saved checkpoint
    if rank == 0:
        if save_handler is not None:
            dirname = save_handler.dirname
        checkpoints = list(os.listdir(dirname))
        assert len(checkpoints) == 1
        for v in [
            "training_checkpoint",
        ]:
            assert any([v in c for c in checkpoints])

    # Check LR scheduling
    assert optimizer.param_groups[0]["lr"] <= lr * gamma ** ((num_iters * num_epochs - 1) // step_size), (
        f"{optimizer.param_groups[0]['lr']} vs {lr * gamma ** ((num_iters * num_epochs - 1) // step_size)}"
    )


def test_asserts_setup_common_training_handlers():
    trainer = Engine(lambda e, b: None)

    with pytest.raises(
        ValueError,
        match=r"If to_save argument is provided then output_path or save_handler arguments should be also defined",
    ):
        setup_common_training_handlers(trainer, to_save={})

    with pytest.raises(ValueError, match=r"Arguments output_path and save_handler are mutually exclusive"):
        setup_common_training_handlers(trainer, to_save={}, output_path="abc", save_handler=lambda c, f, m: None)

    with pytest.warns(UserWarning, match=r"Argument train_sampler is a distributed sampler"):
        train_sampler = MagicMock(spec=DistributedSampler)
        setup_common_training_handlers(trainer, train_sampler=train_sampler)

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match=r"This contrib module requires available GPU"):
            setup_common_training_handlers(trainer, with_gpu_stats=True)

    with pytest.raises(TypeError, match=r"Unhandled type of update_function's output."):
        trainer = Engine(lambda e, b: None)
        setup_common_training_handlers(
            trainer,
            output_names=["loss"],
            with_pbar_on_iters=False,
            with_pbars=False,
            with_gpu_stats=False,
            stop_on_nan=False,
            clear_cuda_cache=False,
        )
        trainer.run([1])


def test_no_warning_with_train_sampler(recwarn):
    from torch.utils.data import RandomSampler

    trainer = Engine(lambda e, b: None)
    train_sampler = RandomSampler([0, 1, 2])
    setup_common_training_handlers(trainer, train_sampler=train_sampler)
    assert len(recwarn) == 0, recwarn.pop()


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" not in os.environ, reason="Should have more than 1 worker")
def test_assert_setup_common_training_handlers_wrong_train_sampler(distributed_context_single_node_gloo):
    trainer = Engine(lambda e, b: None)

    from torch.utils.data.sampler import RandomSampler

    with pytest.raises(TypeError, match=r"Train sampler should be torch DistributedSampler"):
        train_sampler = RandomSampler([0, 1, 2, 3])
        setup_common_training_handlers(trainer, train_sampler)


def test_setup_common_training_handlers(dirname, capsys):
    _test_setup_common_training_handlers(dirname, device="cpu")

    # Check epoch-wise pbar
    captured = capsys.readouterr()
    out = captured.err.split("\r")
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    assert "Epoch" in out[-1] or "Epoch" in out[-2], f"{out[-2]}, {out[-1]}"

    _test_setup_common_training_handlers(dirname, device="cpu", output_transform=lambda loss: [loss])

    # Check epoch-wise pbar
    captured = capsys.readouterr()
    out = captured.err.split("\r")
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    assert "Epoch" in out[-1] or "Epoch" in out[-2], f"{out[-2]}, {out[-1]}"

    _test_setup_common_training_handlers(dirname, device="cpu", output_transform=lambda loss: {"batch_loss": loss})

    # Check epoch-wise pbar
    captured = capsys.readouterr()
    out = captured.err.split("\r")
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    assert "Epoch" in out[-1] or "Epoch" in out[-2], f"{out[-2]}, {out[-1]}"


def test_setup_common_training_handlers_using_save_handler(dirname, capsys):
    save_handler = DiskSaver(dirname=dirname, require_empty=False)
    _test_setup_common_training_handlers(dirname=None, device="cpu", save_handler=save_handler)

    # Check epoch-wise pbar
    captured = capsys.readouterr()
    out = captured.err.split("\r")
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    assert "Epoch" in out[-1] or "Epoch" in out[-2], f"{out[-2]}, {out[-1]}"


def test_save_best_model_by_val_score(dirname):
    acc_scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.5, 0.6, 0.61, 0.7, 0.5]

    def setup_trainer():
        trainer = Engine(lambda e, b: None)
        evaluator = Engine(lambda e, b: None)
        model = DummyModel()

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine):
            evaluator.run([0, 1])

        @evaluator.on(Events.EPOCH_COMPLETED)
        def set_eval_metric(engine):
            acc = acc_scores[trainer.state.epoch - 1]
            engine.state.metrics = {"acc": acc, "loss": 1 - acc}

        return trainer, evaluator, model

    trainer, evaluator, model = setup_trainer()

    save_best_model_by_val_score(dirname, evaluator, model, metric_name="acc", n_saved=2, trainer=trainer)

    trainer.run([0, 1], max_epochs=len(acc_scores))

    assert set(os.listdir(dirname)) == {"best_model_8_val_acc=0.6100.pt", "best_model_9_val_acc=0.7000.pt"}

    for fname in os.listdir(dirname):
        os.unlink(f"{dirname}/{fname}")

    trainer, evaluator, model = setup_trainer()

    save_best_model_by_val_score(
        dirname, evaluator, model, metric_name="loss", n_saved=2, trainer=trainer, score_sign=-1.0
    )

    trainer.run([0, 1], max_epochs=len(acc_scores))

    assert set(os.listdir(dirname)) == {"best_model_8_val_loss=-0.3900.pt", "best_model_9_val_loss=-0.3000.pt"}


def test_gen_save_best_models_by_val_score():
    acc_scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.5, 0.6, 0.61, 0.7, 0.5]
    loss_scores = [0.9, 0.8, 0.7, 0.6, 0.7, 0.5, 0.4, 0.39, 0.3, 0.5]

    def setup_trainer():
        trainer = Engine(lambda e, b: None)
        evaluator = Engine(lambda e, b: None)
        model = DummyModel()

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine):
            evaluator.run([0, 1])

        @evaluator.on(Events.EPOCH_COMPLETED)
        def set_eval_metric(engine):
            acc = acc_scores[trainer.state.epoch - 1]
            loss = loss_scores[trainer.state.epoch - 1]
            engine.state.metrics = {"acc": acc, "loss": loss}

        return trainer, evaluator, model

    trainer, evaluator, model = setup_trainer()

    save_handler = MagicMock()

    gen_save_best_models_by_val_score(
        save_handler, evaluator, {"a": model, "b": model}, metric_name="acc", n_saved=2, trainer=trainer
    )

    trainer.run([0, 1], max_epochs=len(acc_scores))

    assert save_handler.call_count == len(acc_scores) - 2  # 2 score values (0.3 and 0.5) are not the best
    obj_to_save = {"a": model.state_dict(), "b": model.state_dict()}
    save_handler.assert_has_calls(
        [
            call(
                obj_to_save,
                f"best_checkpoint_{e}_val_acc={p:.4f}.pt",
                dict([("basename", "best_checkpoint"), ("score_name", "val_acc"), ("priority", p)]),
            )
            for e, p in zip([1, 2, 3, 4, 6, 7, 8, 9], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.61, 0.7])
        ],
        any_order=True,
    )

    trainer, evaluator, model = setup_trainer()

    save_handler = MagicMock()

    gen_save_best_models_by_val_score(
        save_handler,
        evaluator,
        {"a": model, "b": model},
        metric_name="loss",
        n_saved=2,
        trainer=trainer,
        score_sign=-1.0,
    )

    trainer.run([0, 1], max_epochs=len(acc_scores))

    assert save_handler.call_count == len(acc_scores) - 2  # 2 score values (-0.7 and -0.5) are not the best
    obj_to_save = {"a": model.state_dict(), "b": model.state_dict()}
    save_handler.assert_has_calls(
        [
            call(
                obj_to_save,
                f"best_checkpoint_{e}_val_loss={p:.4f}.pt",
                dict([("basename", "best_checkpoint"), ("score_name", "val_loss"), ("priority", p)]),
            )
            for e, p in zip([1, 2, 3, 4, 6, 7, 8, 9], [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.39, -0.3])
        ],
        any_order=True,
    )


def test_add_early_stopping_by_val_score():
    acc_scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2, 0.1, 0.1, 0.0]

    def setup_trainer():
        trainer = Engine(lambda e, b: None)
        evaluator = Engine(lambda e, b: None)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine):
            evaluator.run([0, 1])

        @evaluator.on(Events.EPOCH_COMPLETED)
        def set_eval_metric(engine):
            acc = acc_scores[trainer.state.epoch - 1]
            engine.state.metrics = {"acc": acc, "loss": 1 - acc}

        return trainer, evaluator

    trainer, evaluator = setup_trainer()

    add_early_stopping_by_val_score(patience=3, evaluator=evaluator, trainer=trainer, metric_name="acc")

    state = trainer.run([0, 1], max_epochs=len(acc_scores))

    assert state.epoch == 7

    trainer, evaluator = setup_trainer()

    add_early_stopping_by_val_score(
        patience=3, evaluator=evaluator, trainer=trainer, metric_name="loss", score_sign=-1.0
    )

    state = trainer.run([0, 1], max_epochs=len(acc_scores))

    assert state.epoch == 7


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(dirname, distributed_context_single_node_nccl):
    local_rank = distributed_context_single_node_nccl["local_rank"]
    device = idist.device()
    _test_setup_common_training_handlers(dirname, device, rank=local_rank, local_rank=local_rank, distributed=True)
    test_add_early_stopping_by_val_score()


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(dirname, distributed_context_single_node_gloo):
    device = idist.device()
    local_rank = distributed_context_single_node_gloo["local_rank"]
    _test_setup_common_training_handlers(dirname, device, rank=local_rank, local_rank=local_rank, distributed=True)
    _test_setup_common_training_handlers(
        dirname, device, rank=local_rank, local_rank=local_rank, distributed=True, lr_scheduler="ignite|LRScheduler"
    )
    _test_setup_common_training_handlers(
        dirname, device, rank=local_rank, local_rank=local_rank, distributed=True, lr_scheduler="ignite"
    )
    test_add_early_stopping_by_val_score()


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(dirname, distributed_context_multi_node_gloo):
    device = idist.device()
    rank = distributed_context_multi_node_gloo["rank"]
    _test_setup_common_training_handlers(dirname, device, rank=rank)
    test_add_early_stopping_by_val_score()


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(dirname, distributed_context_multi_node_nccl):
    local_rank = distributed_context_multi_node_nccl["local_rank"]
    rank = distributed_context_multi_node_nccl["rank"]
    device = idist.device()
    _test_setup_common_training_handlers(dirname, device, rank=rank, local_rank=local_rank, distributed=True)
    test_add_early_stopping_by_val_score()
