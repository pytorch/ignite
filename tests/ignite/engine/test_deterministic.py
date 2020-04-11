import os
import pytest
import random
from unittest.mock import patch

import numpy as np

import torch

from ignite.engine import Engine, Events

from ignite.engine.deterministic import (
    ReproducibleBatchSampler,
    update_dataloader,
    keep_random_state,
    make_deterministic,
)
from ignite.utils import manual_seed

from tests.ignite.engine import BatchChecker, setup_sampler


def test_update_dataloader():
    def _test(sampler_type=None):
        num_epochs = 3
        batch_size = 4
        num_iters = 17
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))
        num_workers = 4

        sampler = setup_sampler(sampler_type, num_iters, batch_size)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            sampler=sampler,
            drop_last=True,
            shuffle=sampler is None,
        )

        torch.manual_seed(12)
        seen_batches = []
        for i in range(num_epochs):
            t = []
            if sampler_type == "distributed":
                sampler.set_epoch(i)
            for b in dataloader:
                t.append(b)
            seen_batches.append(t)

        sampler = setup_sampler(sampler_type, num_iters, batch_size)
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            sampler=sampler,
            drop_last=True,
            shuffle=sampler is None,
        )
        batch_sampler = dataloader.batch_sampler
        new_dataloader = update_dataloader(dataloader, ReproducibleBatchSampler(batch_sampler))

        torch.manual_seed(12)
        new_batches = []
        for i in range(num_epochs):
            t = []
            if sampler_type == "distributed":
                sampler.set_epoch(i)
            for b in new_dataloader:
                t.append(b)
            new_batches.append(t)

        for i in range(num_epochs):
            assert all([(b1 == b2).all() for b1, b2 in zip(seen_batches[i], new_batches[i])])

    _test()
    _test("weighted")
    _test("distributed")


def test_reproducible_batch_sampler_wrong_input():
    with pytest.raises(TypeError, match=r"Argument batch_sampler should be torch.utils.data.sampler.BatchSampler"):
        ReproducibleBatchSampler("abc")


def test_reproducible_batch_sampler():
    import torch
    from torch.utils.data import DataLoader

    data = list(range(100))
    dataloader = DataLoader(data, batch_size=12, num_workers=0, shuffle=True, drop_last=True)

    torch.manual_seed(12 + 0)
    dataloader_ = update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))

    seen_batches = []
    num_epochs = 3
    for i in range(num_epochs):
        t = []
        for b in dataloader_:
            t.append(b)
        seen_batches.append(t)
        torch.manual_seed(12 + i + 1)

    for i in range(num_epochs - 1):
        for j in range(i + 1, num_epochs):
            assert not all([(b1 == b2).all() for b1, b2 in zip(seen_batches[i], seen_batches[j])])

    for resume_epoch in range(num_epochs):
        torch.manual_seed(12 + resume_epoch)
        dataloader_ = update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))
        resumed_seen_batches = []
        for b in dataloader_:
            resumed_seen_batches.append(b)

        assert all([(b1 == b2).all() for b1, b2 in zip(seen_batches[resume_epoch], resumed_seen_batches)])


def _test_keep_random_state(with_numpy):

    manual_seed(54)
    true_values = []
    for _ in range(5):
        t = [
            torch.tensor([random.random()]),
            torch.rand(2),
        ]
        if with_numpy:
            t.append(torch.from_numpy(np.random.rand(2)))
        true_values.append(t)

    @keep_random_state
    def user_handler():
        manual_seed(22)
        _ = [
            random.random(),
            torch.rand(2),
        ]
        if with_numpy:
            _ = np.random.rand(2)

    manual_seed(54)
    res_values = []
    for _ in range(5):
        r = [
            torch.tensor([random.random()]),
            torch.rand(2),
        ]
        if with_numpy:
            r.append(torch.from_numpy(np.random.rand(2)))
        res_values.append(r)
        user_handler()

    for a, b in zip(true_values, res_values):
        for i, j in zip(a, b):
            assert (i == j).all()


def test_keep_random_state():
    _test_keep_random_state(with_numpy=True)


def test_keep_random_state_without_numpy():
    with patch.dict("sys.modules", {"numpy": None}):
        _test_keep_random_state(with_numpy=False)


def test_strict_resume_from_iter():
    def _test(epoch_length=None):

        max_epochs = 5
        num_iters = 21
        torch.manual_seed(0)
        data = torch.randint(0, 1000, size=(num_iters,))
        if epoch_length is None:
            epoch_length = num_iters

        for resume_iteration in range(2, min(num_iters * max_epochs, epoch_length * max_epochs), 4):
            print("\n----", resume_iteration, epoch_length)
            batch_checker = BatchChecker(data, init_counter=resume_iteration)

            def update_fn(_, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_iteration, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)
            make_deterministic(engine)

            @engine.on(Events.EPOCH_COMPLETED)
            def check_iteration(engine):
                assert engine.state.iteration == batch_checker.counter

            resume_state_dict = {
                "iteration": resume_iteration,
                "max_epochs": max_epochs,
                "epoch_length": epoch_length,
            }
            engine.load_state_dict(resume_state_dict)
            engine.run(data)
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(60)
    _test(15)


def test_strict_resume_from_epoch():
    def _test(epoch_length=None):
        max_epochs = 10
        num_iters = 21
        torch.manual_seed(0)
        data = torch.randint(0, 1000, size=(num_iters,))
        if epoch_length is None:
            epoch_length = num_iters

        for resume_epoch in range(1, max_epochs):
            batch_checker = BatchChecker(data, init_counter=resume_epoch * epoch_length)

            def update_fn(_, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_epoch, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)
            make_deterministic(engine)

            resume_state_dict = dict(epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length)
            engine.load_state_dict(resume_state_dict)
            engine.run(data)
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    # _test()
    _test(60)
    _test(15)


def _test_resume_random_dataloader_from_epoch(device, _setup_sampler, sampler_type=None):
    def _test(epoch_length=None):

        max_epochs = 5
        batch_size = 4
        num_iters = 21
        torch.manual_seed(0)
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))

        if epoch_length is None:
            epoch_length = num_iters

        for resume_epoch in range(1, max_epochs):

            for num_workers in [0, 4]:
                sampler = _setup_sampler(sampler_type, num_iters, batch_size)
                orig_dataloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in device,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )

                seen_batchs = []

                def update_fn(_, batch):
                    batch_to_device = batch.to(device)
                    seen_batchs.append(batch)

                engine = Engine(update_fn)
                make_deterministic(engine, seed=12)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch - 1)

                engine.run(
                    orig_dataloader, max_epochs=max_epochs, epoch_length=epoch_length,
                )

                batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

                sampler = _setup_sampler(sampler_type, num_iters, batch_size)
                resume_dataloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in device,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )

                def update_fn(_, batch):
                    batch_to_device = batch.to(device)
                    assert batch_checker.check(batch), "{} {} | {}: {} vs {}".format(
                        num_workers, resume_epoch, batch_checker.counter, batch_checker.true_batch, batch
                    )

                engine = Engine(update_fn)
                make_deterministic(engine, seed=12)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch - 1)

                resume_state_dict = dict(epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length)
                engine.load_state_dict(resume_state_dict)
                torch.manual_seed(1)
                engine.run(resume_dataloader)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs

    _test()
    if sampler_type != "distributed":
        _test(60)
        _test(15)


def test_resume_random_dataloader_from_epoch():
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler)
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler, sampler_type="weighted")
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler, sampler_type="distributed")


class AugmentedData:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        dp = self.data[i]
        r = torch.randint_like(dp, 0, 100)
        return dp + r

    def __len__(self):
        return len(self.data)


def _test_resume_random_dataloader_from_iter(device, _setup_sampler, sampler_type=None):
    def _test(epoch_length=None):
        max_epochs = 3
        batch_size = 4
        num_iters = 17
        torch.manual_seed(0)
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))

        if epoch_length is None:
            epoch_length = num_iters

        for resume_iteration in range(2, min(num_iters * max_epochs, epoch_length * max_epochs), 7):

            for num_workers in [0, 4]:

                sampler = _setup_sampler(sampler_type, num_iters, batch_size)
                orig_dataloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in device,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )
                seen_batchs = []

                def update_fn(engine, batch):
                    batch_to_device = batch.to(device)
                    seen_batchs.append(batch)

                engine = Engine(update_fn)
                make_deterministic(engine, seed=12)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch)

                torch.manual_seed(12)
                engine.run(
                    orig_dataloader, max_epochs=max_epochs, epoch_length=epoch_length,
                )

                batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration)

                sampler = _setup_sampler(sampler_type, num_iters, batch_size)
                resume_dataloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in device,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )

                def update_fn(engine, batch):
                    batch_to_device = batch.to(device)
                    assert batch_checker.check(batch), "{} {} | {}: {} vs {}".format(
                        num_workers, resume_iteration, batch_checker.counter, batch_checker.true_batch, batch
                    )

                engine = Engine(update_fn)
                make_deterministic(engine, seed=12)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch)

                resume_state_dict = dict(iteration=resume_iteration, max_epochs=max_epochs, epoch_length=epoch_length)
                engine.load_state_dict(resume_state_dict)
                torch.manual_seed(12)
                engine.run(resume_dataloader)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs, "{}, {} | {} vs {}".format(
                    num_workers, resume_iteration, engine.state.iteration, epoch_length * max_epochs
                )

    _test()
    if sampler_type != "distributed":
        _test(40)
        _test(11)


def test_resume_random_dataloader_from_iter():
    _test_resume_random_dataloader_from_iter("cpu", setup_sampler)
    _test_resume_random_dataloader_from_iter("cpu", setup_sampler, sampler_type="weighted")
    _test_resume_random_dataloader_from_iter("cpu", setup_sampler, sampler_type="distributed")


def _test_resume_random_data_iterator_from_epoch(device):
    def _test(epoch_length=None):
        max_epochs = 5
        batch_size = 4
        num_iters = 21

        def infinite_data_iterator():
            while True:
                for _ in range(num_iters):
                    data = torch.randint(0, 1000, size=(batch_size,), device=device)
                    yield data

        if epoch_length is None:
            epoch_length = num_iters

        for resume_epoch in range(1, max_epochs):
            seen_batchs = []

            def update_fn(engine, batch):
                # if there is a random op when using data batch etc, we can not resume correctly
                # torch.rand(1)
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            make_deterministic(engine, seed=12)

            engine.run(
                infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length,
            )

            batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_epoch, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)
            make_deterministic(engine, seed=12)

            resume_state_dict = dict(epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length)
            engine.load_state_dict(resume_state_dict)
            engine.run(infinite_data_iterator())
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(60)
    _test(15)


def test_resume_random_data_iterator_from_epoch():
    _test_resume_random_data_iterator_from_epoch("cpu")


def _test_resume_random_data_iterator_from_iter(device):
    def _test(epoch_length=None):
        max_epochs = 3
        batch_size = 4
        num_iters = 17

        def infinite_data_iterator():
            while True:
                for _ in range(num_iters):
                    data = torch.randint(0, 1000, size=(batch_size,), device=device)
                    yield data

        if epoch_length is None:
            epoch_length = num_iters

        for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs), 7):

            seen_batchs = []

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            make_deterministic(engine, seed=12)

            torch.manual_seed(12)
            engine.run(
                infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length,
            )

            batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_iteration, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)
            make_deterministic(engine, seed=12)

            resume_state_dict = dict(iteration=resume_iteration, max_epochs=max_epochs, epoch_length=epoch_length)
            engine.load_state_dict(resume_state_dict)
            torch.manual_seed(12)
            engine.run(infinite_data_iterator())
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs, "{} | {} vs {}".format(
                resume_iteration, engine.state.iteration, epoch_length * max_epochs
            )

    _test()
    _test(50)
    _test(11)


def test_resume_random_data_iterator_from_iter():
    _test_resume_random_data_iterator_from_iter("cpu")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl, setup_sampler):
    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = "cpu"
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl["local_rank"])
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler)


def test_concepts_snippet_resume():
    import torch
    from torch.utils.data import DataLoader
    from ignite.engine import Engine, Events
    from ignite.engine.deterministic import make_deterministic

    seen_batches = []

    def random_train_data_loader(size):
        data = torch.arange(0, size)
        return DataLoader(data, batch_size=4, shuffle=True)

    def print_train_data(engine, batch):
        i = engine.state.iteration
        e = engine.state.epoch
        print("train", e, i, batch.tolist())
        seen_batches.append(batch)

    trainer = Engine(print_train_data)
    make_deterministic(trainer, seed=15)

    print("Original Run")
    trainer.run(random_train_data_loader(40), max_epochs=2, epoch_length=5)

    original_batches = list(seen_batches)
    seen_batches = []

    print("Resumed Run")
    trainer.load_state_dict({"epoch": 1, "seed": 7, "epoch_length": 5, "max_epochs": 2})
    trainer.run(random_train_data_loader(40))

    resumed_batches = list(seen_batches)
    seen_batches = []
    for b1, b2 in zip(original_batches[5:], resumed_batches):
        assert (b1 == b2).all()


def test_concepts_snippet_warning():
    def random_train_data_generator():
        while True:
            yield torch.randint(0, 100, size=(1,))

    def print_train_data(engine, batch):
        i = engine.state.iteration
        e = engine.state.epoch
        print("train", e, i, batch.tolist())

    trainer = Engine(print_train_data)
    make_deterministic(trainer, seed=15)

    @trainer.on(Events.ITERATION_COMPLETED(every=3))
    def user_handler(_):
        # handler synchronizes the random state
        torch.manual_seed(12)
        a = torch.rand(1)

    trainer.run(random_train_data_generator(), max_epochs=3, epoch_length=5)
