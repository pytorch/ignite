import os
import random
import sys
from collections.abc import Mapping
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from packaging.version import Version
from torch.optim import SGD
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

import ignite.distributed as idist
from ignite.engine import Events
from ignite.engine.deterministic import (
    _set_rng_states,
    DeterministicEngine,
    keep_random_state,
    ReproducibleBatchSampler,
    update_dataloader,
)
from ignite.utils import manual_seed
from tests.ignite.engine import BatchChecker, setup_sampler


def test_dengine_setup_seed_div_by_zero():
    with pytest.raises(ValueError, match=r"iter_counter should be positive value"):
        DeterministicEngine(lambda e, b: None)._setup_seed(iter_counter=0)


def test_update_dataloader():
    def _test(sampler_type=None):
        num_epochs = 3
        total_batch_size = 4
        num_iters = 17
        data = torch.randint(0, 1000, size=(num_iters * total_batch_size,))
        num_workers = 2

        sampler, batch_size = setup_sampler(sampler_type, num_iters, total_batch_size)
        dataloader = DataLoader(
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

        sampler, batch_size = setup_sampler(sampler_type, num_iters, total_batch_size)
        dataloader = DataLoader(
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
            batch_checker = BatchChecker(data, init_counter=resume_iteration)

            def update_fn(_, batch):
                assert batch_checker.check(
                    batch
                ), f"{resume_iteration} | {batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

            engine = DeterministicEngine(update_fn)

            @engine.on(Events.EPOCH_COMPLETED)
            def check_iteration(_):
                assert engine.state.iteration == batch_checker.counter

            resume_state_dict = dict(
                iteration=resume_iteration, max_epochs=max_epochs, epoch_length=epoch_length, rng_states=None
            )
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
                assert batch_checker.check(
                    batch
                ), f"{resume_epoch} | {batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

            engine = DeterministicEngine(update_fn)

            resume_state_dict = dict(
                epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length, rng_states=None
            )
            engine.load_state_dict(resume_state_dict)
            engine.run(data)
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(60)
    _test(15)


def _test_resume_random_dataloader_from_epoch(device, _setup_sampler, sampler_type=None):
    def _test(epoch_length=None):
        max_epochs = 5
        total_batch_size = 4
        num_iters = 21
        torch.manual_seed(0)
        data = torch.randint(0, 1000, size=(num_iters * total_batch_size,))

        if epoch_length is None:
            epoch_length = num_iters

        for resume_epoch in range(1, max_epochs, 2):
            for num_workers in [0, 2]:
                sampler, batch_size = _setup_sampler(sampler_type, num_iters, total_batch_size)

                orig_dataloader = DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in torch.device(device).type,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )

                seen_batchs = []

                def update_fn(_, batch):
                    batch_to_device = batch.to(device)
                    seen_batchs.append(batch)

                engine = DeterministicEngine(update_fn)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch - 1)

                torch.manual_seed(87)
                engine.run(orig_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

                batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

                sampler, batch_size = _setup_sampler(sampler_type, num_iters, total_batch_size)
                resume_dataloader = DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in torch.device(device).type,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )

                def update_fn(_, batch):
                    batch_to_device = batch.to(device)
                    assert batch_checker.check(
                        batch
                    ), f"{num_workers} {resume_epoch} | {batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

                engine = DeterministicEngine(update_fn)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch - 1)

                resume_state_dict = dict(
                    epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length, rng_states=None
                )
                engine.load_state_dict(resume_state_dict)
                torch.manual_seed(87)
                engine.run(resume_dataloader)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs

    _test()
    if sampler_type != "distributed":
        _test(60)
        _test(15)


@pytest.mark.skipif("win" in sys.platform, reason="Skip extremely slow test on Windows/MacOSX")
def test_resume_random_dataloader_from_epoch():
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler)
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler, sampler_type="weighted")
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler, sampler_type="distributed")


class AugmentedData:
    def __init__(self, data, enabled=True):
        self.data = data
        self.enabled = enabled

    def __getitem__(self, i):
        dp = self.data[i]
        r = torch.randint_like(dp, -100, 100) if self.enabled else 0.0
        return dp + r * 0.01

    def __len__(self):
        return len(self.data)


def _test_resume_random_dataloader_from_iter(device, _setup_sampler, sampler_type=None):
    def _test(epoch_length=None):
        max_epochs = 3
        total_batch_size = 4
        num_iters = 17
        torch.manual_seed(0)
        data = torch.randint(0, 1000, size=(num_iters * total_batch_size,))

        if epoch_length is None:
            epoch_length = num_iters

        for resume_iteration in range(2, min(num_iters * max_epochs, epoch_length * max_epochs), 13):
            for num_workers in [0, 2]:
                sampler, batch_size = _setup_sampler(sampler_type, num_iters, total_batch_size)
                orig_dataloader = DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in torch.device(device).type,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )
                seen_batchs = []

                def update_fn(_, batch):
                    batch_to_device = batch.to(device)
                    seen_batchs.append(batch)

                engine = DeterministicEngine(update_fn)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch)

                torch.manual_seed(12)
                engine.run(orig_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

                batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration)

                sampler, batch_size = _setup_sampler(sampler_type, num_iters, total_batch_size)
                resume_dataloader = DataLoader(
                    data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory="cuda" in torch.device(device).type,
                    sampler=sampler,
                    drop_last=True,
                    shuffle=sampler is None,
                )

                def update_fn(_, batch):
                    batch_to_device = batch.to(device)
                    cfg_msg = f"{num_workers} {resume_iteration}"
                    assert batch_checker.check(
                        batch
                    ), f"{cfg_msg} | {batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

                engine = DeterministicEngine(update_fn)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch)

                resume_state_dict = dict(
                    iteration=resume_iteration, max_epochs=max_epochs, epoch_length=epoch_length, rng_states=None
                )
                engine.load_state_dict(resume_state_dict)
                torch.manual_seed(12)
                engine.run(resume_dataloader)
                assert engine.state.epoch == max_epochs
                assert (
                    engine.state.iteration == epoch_length * max_epochs
                ), f"{num_workers}, {resume_iteration} | {engine.state.iteration} vs {epoch_length * max_epochs}"

    _test()
    if sampler_type != "distributed":
        _test(40)
        _test(11)


@pytest.mark.skipif("win" in sys.platform, reason="Skip extremely slow test on Windows/MacOSX")
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

            def update_fn(_, batch):
                # if there is a random op when using data batch etc, we can not resume correctly
                # torch.rand(1)
                seen_batchs.append(batch)

            engine = DeterministicEngine(update_fn)
            torch.manual_seed(121)
            engine.run(infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length)

            batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

            def update_fn(_, batch):
                assert batch_checker.check(
                    batch
                ), f"{resume_epoch} | {batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

            engine = DeterministicEngine(update_fn)

            resume_state_dict = dict(
                epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length, rng_states=None
            )
            engine.load_state_dict(resume_state_dict)
            torch.manual_seed(121)
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

            def update_fn(_, batch):
                seen_batchs.append(batch)

            engine = DeterministicEngine(update_fn)

            torch.manual_seed(24)
            engine.run(infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length)

            batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration)

            def update_fn(_, batch):
                assert batch_checker.check(
                    batch
                ), f"{resume_iteration} | {batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

            engine = DeterministicEngine(update_fn)

            resume_state_dict = dict(
                iteration=resume_iteration, max_epochs=max_epochs, epoch_length=epoch_length, rng_states=None
            )
            engine.load_state_dict(resume_state_dict)
            torch.manual_seed(24)
            engine.run(infinite_data_iterator())
            assert engine.state.epoch == max_epochs
            assert (
                engine.state.iteration == epoch_length * max_epochs
            ), f"{resume_iteration} | {engine.state.iteration} vs {epoch_length * max_epochs}"

    _test()
    _test(50)
    _test(11)


def test_resume_random_data_iterator_from_iter():
    _test_resume_random_data_iterator_from_iter("cpu")


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_resume_random_dataloader_from_iter(device, setup_sampler, sampler_type="distributed")
    _test_resume_random_dataloader_from_epoch(device, setup_sampler, sampler_type="distributed")


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_resume_random_dataloader_from_iter(device, setup_sampler, sampler_type="distributed")
    _test_resume_random_dataloader_from_epoch(device, setup_sampler, sampler_type="distributed")


@pytest.mark.xfail
@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_resume_random_dataloader_from_iter(device, setup_sampler, sampler_type="distributed")
    _test_resume_random_dataloader_from_epoch(device, setup_sampler, sampler_type="distributed")


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_resume_random_dataloader_from_iter(device, setup_sampler, sampler_type="distributed")
    _test_resume_random_dataloader_from_epoch(device, setup_sampler, sampler_type="distributed")


def test_concepts_snippet_resume():
    # Commented imports required in the snippet
    # import torch
    # from torch.utils.data import DataLoader

    # from ignite.engine import DeterministicEngine
    # from ignite.utils import manual_seed

    seen_batches = []
    manual_seed(seed=15)

    def random_train_data_loader(size):
        data = torch.arange(0, size)
        return DataLoader(data, batch_size=4, shuffle=True)

    def print_train_data(engine, batch):
        i = engine.state.iteration
        e = engine.state.epoch
        print("train", e, i, batch.tolist())
        seen_batches.append(batch)

    trainer = DeterministicEngine(print_train_data)

    print("Original Run")
    manual_seed(56)
    trainer.run(random_train_data_loader(40), max_epochs=2, epoch_length=5)

    original_batches = list(seen_batches)
    seen_batches = []

    print("Resumed Run")
    trainer.load_state_dict({"epoch": 1, "epoch_length": 5, "max_epochs": 2, "rng_states": None})
    manual_seed(56)
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

    trainer = DeterministicEngine(print_train_data)

    @trainer.on(Events.ITERATION_COMPLETED(every=3))
    def user_handler(_):
        # handler synchronizes the random state
        torch.manual_seed(12)
        a = torch.rand(1)

    trainer.run(random_train_data_generator(), max_epochs=3, epoch_length=5)


def _test_gradients_on_resume(
    dirname, device, with_dropout=True, with_dataaugs=True, data_size=24, batch_size=4, save_iter=None, save_epoch=None
):
    debug = False

    def random_train_data_loader(size):
        d = AugmentedData(torch.rand(size, 3, 32, 32), enabled=with_dataaugs)
        return DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=2)

    def _train(save_iter=None, save_epoch=None, sd=None):
        w_norms = []
        grad_norms = []
        data = []
        chkpt = []

        manual_seed(12)
        arch = [
            nn.Conv2d(3, 10, 3),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
        ]
        if with_dropout:
            arch.insert(2, nn.Dropout2d())
            arch.insert(-2, nn.Dropout())

        model = nn.Sequential(*arch).to(device)
        opt = SGD(model.parameters(), lr=0.001)

        def proc_fn(e, b):
            from ignite.engine.deterministic import _get_rng_states, _repr_rng_state

            s = _repr_rng_state(_get_rng_states())
            model.train()
            opt.zero_grad()
            y = model(b.to(device))
            y.sum().backward()
            opt.step()
            if debug:
                print(
                    trainer.state.iteration, trainer.state.epoch, "proc_fn - b.shape", b.shape, torch.norm(y).item(), s
                )

        trainer = DeterministicEngine(proc_fn)

        if save_iter is not None:
            ev = Events.ITERATION_COMPLETED(once=save_iter)
        elif save_epoch is not None:
            ev = Events.EPOCH_COMPLETED(once=save_epoch)
            save_iter = save_epoch * (data_size // batch_size)

        @trainer.on(ev)
        def save_chkpt(_):
            if debug:
                print(trainer.state.iteration, "save_chkpt")
            fp = dirname / "test.pt"
            from ignite.engine.deterministic import _repr_rng_state

            tsd = trainer.state_dict()
            if debug:
                print("->", _repr_rng_state(tsd["rng_states"]))
            torch.save([model.state_dict(), opt.state_dict(), tsd], fp)
            chkpt.append(fp)

        def log_event_filter(_, event):
            if (event // save_iter == 1) and 1 <= (event % save_iter) <= 5:
                return True
            return False

        @trainer.on(Events.ITERATION_COMPLETED(event_filter=log_event_filter))
        def write_data_grads_weights(e):
            x = e.state.batch
            i = e.state.iteration
            data.append([i, x.mean().item(), x.std().item()])

            total = [0.0, 0.0]
            out1 = []
            out2 = []
            for p in model.parameters():
                n1 = torch.norm(p).item()
                n2 = torch.norm(p.grad).item()
                out1.append(n1)
                out2.append(n2)
                total[0] += n1
                total[1] += n2
            w_norms.append([i, total[0]] + out1)
            grad_norms.append([i, total[1]] + out2)

        if sd is not None:
            if Version(torch.__version__) >= Version("1.13.0"):
                kwargs = {"weights_only": False}
            else:
                kwargs = {}
            sd = torch.load(sd, **kwargs)
            model.load_state_dict(sd[0])
            opt.load_state_dict(sd[1])
            from ignite.engine.deterministic import _repr_rng_state

            if debug:
                print("-->", _repr_rng_state(sd[2]["rng_states"]))
            trainer.load_state_dict(sd[2])

        manual_seed(32)
        trainer.run(random_train_data_loader(size=data_size), max_epochs=5)
        return {"sd": chkpt, "data": data, "grads": grad_norms, "weights": w_norms}

    out_original = _train(save_iter=save_iter, save_epoch=save_epoch)
    assert len(out_original["sd"]) > 0

    out_resumed = _train(save_iter=save_iter, save_epoch=save_epoch, sd=out_original["sd"][0])

    if debug:
        print("Original:")
        print(" data:", out_original["data"])
        print("grads:", out_original["grads"])
        print("    W:", out_original["weights"])
        print("Resume:")
        print(" data:", out_resumed["data"])
        print("grads:", out_resumed["grads"])
        print("    W:", out_resumed["weights"])

    # check data:
    for d1, d2 in zip(out_original["data"], out_resumed["data"]):
        assert d1 == d2

    # check grads:
    for d1, d2 in zip(out_original["grads"], out_resumed["grads"]):
        assert d1 == d2

    # check weights:
    for d1, d2 in zip(out_original["weights"], out_resumed["weights"]):
        assert d1 == d2


def test_gradients_on_resume_cpu(dirname):
    with pytest.raises(AssertionError):
        _test_gradients_on_resume(dirname, "cpu", with_dataaugs=True, save_iter=25)
    _test_gradients_on_resume(dirname, "cpu", with_dataaugs=False, save_iter=25)
    # resume from epoch
    _test_gradients_on_resume(dirname, "cpu", with_dataaugs=True, save_epoch=3)
    _test_gradients_on_resume(dirname, "cpu", with_dataaugs=False, save_epoch=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_gradients_on_resume_on_cuda(dirname):
    with pytest.raises(AssertionError):
        _test_gradients_on_resume(dirname, "cuda", with_dataaugs=True, save_iter=25)
    with pytest.raises(AssertionError):
        _test_gradients_on_resume(dirname, "cuda", with_dataaugs=False, save_iter=25)
    # resume from epoch
    _test_gradients_on_resume(dirname, "cuda", with_dataaugs=True, save_epoch=3)
    _test_gradients_on_resume(dirname, "cuda", with_dataaugs=False, save_epoch=3)


def test_engine_with_dataloader_no_auto_batching():
    # tests https://github.com/pytorch/ignite/issues/941

    data = torch.rand(64, 4, 10)
    data_loader = DataLoader(
        data, batch_size=None, sampler=BatchSampler(RandomSampler(data), batch_size=8, drop_last=True)
    )

    counter = [0]

    def foo(e, b):
        print(f"{e.state.epoch}-{e.state.iteration}: {b}")
        counter[0] += 1

    engine = DeterministicEngine(foo)
    engine.run(data_loader, epoch_length=10, max_epochs=5)

    assert counter[0] == 50


def test_run_finite_iterator_no_epoch_length():
    # FR: https://github.com/pytorch/ignite/issues/871
    unknown_size = 11

    def finite_unk_size_data_iter():
        for i in range(unknown_size):
            yield i

    bc = BatchChecker(data=list(range(unknown_size)))

    engine = DeterministicEngine(lambda e, b: bc.check(b))

    @engine.on(Events.DATALOADER_STOP_ITERATION)
    def restart_iter():
        engine.state.dataloader = finite_unk_size_data_iter()

    data_iter = finite_unk_size_data_iter()
    engine.run(data_iter, max_epochs=5)

    assert engine.state.epoch == 5
    assert engine.state.iteration == unknown_size * 5


class OldDataLoader(DataLoader):
    def __init__(self, dl, *args, **kwargs):
        self.dl = dl
        self.sampler = self.dl.sampler
        self.batch_sampler = self.dl.batch_sampler

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        return iter(self.dl)


def test_dataloader_no_dataset_kind():
    # tests issue : https://github.com/pytorch/ignite/issues/1022

    engine = DeterministicEngine(lambda e, b: None)

    data = torch.randint(0, 1000, size=(100 * 4,))
    dataloader = DataLoader(data, batch_size=4)
    dataloader = OldDataLoader(dataloader)

    engine.run(dataloader)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test__set_rng_states_cuda():
    # Checks https://github.com/pytorch/ignite/issues/2076

    rng_states = [random.getstate(), torch.get_rng_state().cuda(), np.random.get_state()]
    _set_rng_states(rng_states)
    assert rng_states[1].device.type == "cpu"


def test_engine_no_data_asserts():
    trainer = DeterministicEngine(lambda e, b: None)

    with pytest.raises(ValueError, match=r"Deterministic engine does not support the option of data=None"):
        trainer.run(max_epochs=10, epoch_length=10)


def test_state_dict():
    engine = DeterministicEngine(lambda e, b: 1)
    sd = engine.state_dict()
    assert isinstance(sd, Mapping) and len(sd) == 4
    assert "iteration" in sd and sd["iteration"] == 0
    assert "max_epochs" in sd and sd["max_epochs"] is None
    assert "epoch_length" in sd and sd["epoch_length"] is None
    assert "rng_states" in sd and sd["rng_states"] is not None
