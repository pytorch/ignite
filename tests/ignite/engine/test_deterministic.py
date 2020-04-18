import os
import pytest
import random
from unittest.mock import patch

import numpy as np

import torch
import torch.nn as nn

from ignite.engine.deterministic import (
    ReproducibleBatchSampler,
    update_dataloader,
    keep_random_state,
    DeterministicEngine,
)

from ignite.engine import Events
from ignite.utils import manual_seed

from tests.ignite.engine import setup_sampler, BatchChecker


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

            engine = DeterministicEngine(update_fn)

            @engine.on(Events.EPOCH_COMPLETED)
            def check_iteration(engine):
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
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_epoch, batch_checker.counter, batch_checker.true_batch, batch
                )

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

                engine = DeterministicEngine(update_fn)

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch - 1)

                torch.manual_seed(87)
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

                engine = DeterministicEngine(update_fn)

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

            engine = DeterministicEngine(update_fn)
            torch.manual_seed(121)
            engine.run(
                infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length,
            )

            batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_epoch, batch_checker.counter, batch_checker.true_batch, batch
                )

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

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = DeterministicEngine(update_fn)

            torch.manual_seed(24)
            engine.run(
                infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length,
            )

            batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_iteration, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = DeterministicEngine(update_fn)

            resume_state_dict = dict(
                iteration=resume_iteration, max_epochs=max_epochs, epoch_length=epoch_length, rng_states=None
            )
            engine.load_state_dict(resume_state_dict)
            torch.manual_seed(24)
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
    from ignite.engine import DeterministicEngine, Events
    from ignite.utils import manual_seed

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


def _test_gradients_on_resume(dirname, device, with_dropout=True, with_dataaugs=True):

    debug = False

    from torch.utils.data import DataLoader
    from torch.optim import SGD

    def random_train_data_loader(size):
        d = AugmentedData(torch.rand(size, 3, 32, 32), enabled=with_dataaugs)
        return DataLoader(d, batch_size=4, shuffle=True, num_workers=4)

    def _train(save_iter, sd=None):
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
            from ignite.engine.deterministic import _repr_rng_state, _get_rng_states

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

        @trainer.on(Events.ITERATION_COMPLETED(once=save_iter))
        def save_chkpt(_):
            if debug:
                print(trainer.state.iteration, "save_chkpt")
            fp = os.path.join(dirname, "test.pt")
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
            sd = torch.load(sd)
            model.load_state_dict(sd[0])
            opt.load_state_dict(sd[1])
            from ignite.engine.deterministic import _repr_rng_state

            if debug:
                print("-->", _repr_rng_state(sd[2]["rng_states"]))
            trainer.load_state_dict(sd[2])

        manual_seed(32)
        trainer.run(random_train_data_loader(size=24), max_epochs=5)
        return {"sd": chkpt, "data": data, "grads": grad_norms, "weights": w_norms}

    save_iter = 25
    out_original = _train(save_iter=save_iter)
    assert len(out_original["sd"]) > 0

    out_resumed = _train(save_iter=save_iter, sd=out_original["sd"][0])

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
        _test_gradients_on_resume(dirname, "cpu", with_dataaugs=True)
    _test_gradients_on_resume(dirname, "cpu", with_dataaugs=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_gradients_on_resume_gpu(dirname):
    with pytest.raises(AssertionError):
        _test_gradients_on_resume(dirname, "cuda", with_dataaugs=True)
    _test_gradients_on_resume(dirname, "cuda", with_dataaugs=False)
