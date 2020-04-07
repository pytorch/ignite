import os
import pytest

from collections.abc import Mapping

import torch

from ignite.engine import Engine, State, Events


def test_state_dict():
    engine = Engine(lambda e, b: 1)
    sd = engine.state_dict()
    assert isinstance(sd, Mapping) and len(sd) == 0

    def _test(state):
        engine.state = state
        sd = engine.state_dict()
        assert isinstance(sd, Mapping) and len(sd) == len(engine._state_dict_all_req_keys) + 1
        assert sd["iteration"] == engine.state.iteration
        assert sd["epoch_length"] == engine.state.epoch_length
        assert sd["max_epochs"] == engine.state.max_epochs

    _test(State(iteration=500, epoch_length=1000, max_epochs=100))
    _test(State(epoch=5, epoch_length=1000, max_epochs=100))


def test_state_dict_integration():
    engine = Engine(lambda e, b: 1)
    data = range(100)
    engine.run(data, max_epochs=10)
    sd = engine.state_dict()
    assert isinstance(sd, Mapping) and len(sd) == len(engine._state_dict_all_req_keys) + 1
    assert sd["iteration"] == engine.state.iteration == 10 * 100
    assert sd["epoch_length"] == engine.state.epoch_length == 100
    assert sd["max_epochs"] == engine.state.max_epochs == 10


def test_load_state_dict_asserts():
    engine = Engine(lambda e, b: 1)

    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary"):
        engine.load_state_dict("123")

    with pytest.raises(ValueError, match=r"is absent in provided state_dict"):
        engine.load_state_dict({})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        engine.load_state_dict({"max_epochs": 100, "epoch_length": 120})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        engine.load_state_dict({"max_epochs": 100, "epoch_length": 120, "iteration": 12, "epoch": 123})


def test_load_state_dict():
    engine = Engine(lambda e, b: 1)

    def _test(sd):
        engine.load_state_dict(sd)
        if "iteration" in sd:
            assert sd["iteration"] == engine.state.iteration + 1
        elif "epoch" in sd:
            assert sd["epoch"] == engine.state.epoch + 1
        assert sd["epoch_length"] == engine.state.epoch_length
        assert sd["max_epochs"] == engine.state.max_epochs

    _test({"max_epochs": 100, "epoch_length": 120, "iteration": 123})
    _test({"max_epochs": 100, "epoch_length": 120, "epoch": 5})


def test_load_state_dict_integration(counter_factory):
    engine = Engine(lambda e, b: 1)

    state_dict = {"max_epochs": 100, "epoch_length": 120, "epoch": 5}

    engine.load_state_dict(state_dict)
    engine.add_event_handler(Events.ITERATION_COMPLETED, counter_factory("iter", 4 * 120 + 1))
    engine.add_event_handler(Events.EPOCH_COMPLETED, counter_factory("epoch", 5))
    data = range(120)
    engine.run(data)


def test_load_state_dict_with_params_overriding_integration(counter_factory):

    state_dict = {"max_epochs": 100, "epoch_length": 120, "epoch": 5}
    data = range(120)

    # Override max_epochs
    new_max_epochs = 10
    engine = Engine(lambda e, b: 1)
    engine.load_state_dict(state_dict)
    state = engine.run(data, max_epochs=new_max_epochs)
    assert state.max_epochs == new_max_epochs
    assert state.iteration == state_dict["epoch_length"] * new_max_epochs
    assert state.epoch == new_max_epochs

    with pytest.raises(ValueError, match=r"Argument max_epochs should be larger than the start epoch"):
        engine.load_state_dict(state_dict)
        engine.run(data, max_epochs=3)

    # Override epoch_length
    with pytest.raises(ValueError, match=r"Argument epoch_length should be same as in the state"):
        engine.load_state_dict(state_dict)
        engine.run(data, epoch_length=90)


class BatchChecker:
    def __init__(self, data, init_counter=0):
        self.counter = init_counter
        self.data = data
        self.true_batch = None

    def check(self, batch):
        self.true_batch = self.data[self.counter % len(self.data)]
        self.counter += 1
        return (self.true_batch == batch).all()


def test_epoch_length():
    def _test(data, max_epochs, num_iters):

        batch_checker = BatchChecker(data)

        def update_fn(_, batch):
            assert batch_checker.check(batch), "{}: {} vs {}".format(
                batch_checker.counter, batch_checker.true_batch, batch
            )

        engine = Engine(update_fn)
        engine.run(data, max_epochs=max_epochs, epoch_length=num_iters)
        if num_iters is None:
            num_iters = len(data)
        assert engine.state.iteration == num_iters * max_epochs
        assert engine.state.epoch == max_epochs

    def _test_as_iter(data, max_epochs, num_iters):

        batch_checker = BatchChecker(data)

        def update_fn(_, batch):
            assert batch_checker.check(batch), "{}: {} vs {}".format(
                batch_checker.counter, batch_checker.true_batch, batch
            )

        engine = Engine(update_fn)
        engine.run(iter(data), max_epochs=max_epochs, epoch_length=num_iters)
        if num_iters is None:
            num_iters = len(data)
        assert engine.state.iteration == num_iters * max_epochs
        assert engine.state.epoch == max_epochs

    max_epochs = 10
    num_iters = 21
    data = torch.randint(0, 1000, size=(num_iters,))
    _test(data, max_epochs, num_iters=None)
    _test(data, max_epochs, num_iters)
    _test(data, max_epochs, num_iters // 2)
    _test(data, max_epochs, num_iters * 2)

    _test_as_iter(data, 1, num_iters)
    _test_as_iter(data, 2, num_iters // 2)


@pytest.mark.skip("There is no more resume from iteration")
def test_strict_resume_from_iter():
    def _test(epoch_length=None):

        max_epochs = 5
        num_iters = 21
        data = torch.randint(0, 1000, size=(num_iters,))
        if epoch_length is None:
            epoch_length = num_iters

        for resume_iteration in range(2, min(num_iters * max_epochs, epoch_length * max_epochs), 4):
            batch_checker = BatchChecker(data, init_counter=resume_iteration - 1)

            def update_fn(_, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_iteration, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)

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
        data = torch.randint(0, 1000, size=(num_iters,))
        if epoch_length is None:
            epoch_length = num_iters

        for resume_epoch in range(2, max_epochs):
            batch_checker = BatchChecker(data, init_counter=(resume_epoch - 1) * epoch_length)

            def update_fn(_, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_epoch, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)
            resume_state_dict = dict(epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length)
            engine.load_state_dict(resume_state_dict)
            engine.run(data)
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    _test()
    # Below tests wont work as there is no more
    # _test(60)
    # _test(15)


def _test_resume_random_dataloader_from_epoch(device, _setup_sampler, sampler_type=None):
    def _test(epoch_length=None):

        max_epochs = 5
        batch_size = 4
        num_iters = 21
        torch.manual_seed(0)
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))

        if epoch_length is None:
            epoch_length = num_iters

        for resume_epoch in range(2, max_epochs + 1):

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

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch - 1)

                @engine.on(Events.EPOCH_STARTED(once=resume_epoch))
                def sync_seed(e):
                    e.sync_dataflow()

                torch.manual_seed(1)
                engine.run(
                    orig_dataloader, max_epochs=max_epochs, epoch_length=epoch_length,
                )

                batch_checker = BatchChecker(seen_batchs, init_counter=(resume_epoch - 1) * epoch_length)

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


def test_resume_random_dataloader_from_epoch(setup_sampler_fn):
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler_fn)
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler_fn, sampler_type="weighted")
    _test_resume_random_dataloader_from_epoch("cpu", setup_sampler_fn, sampler_type="distributed")


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

    torch.manual_seed(0)

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

                if sampler_type == "distributed":

                    @engine.on(Events.EPOCH_STARTED)
                    def _(engine):
                        sampler.set_epoch(engine.state.epoch)

                @engine.on(Events.GET_BATCH_STARTED(once=resume_iteration - 1))
                def sync_seed(e):
                    if sampler_type == "distributed":
                        sampler.set_epoch(engine.state.epoch)
                    e.sync_dataflow()

                torch.manual_seed(12)
                engine.run(
                    orig_dataloader, max_epochs=max_epochs, epoch_length=epoch_length,
                )

                batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration - 1)

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


def test_resume_random_dataloader_from_iter(setup_sampler_fn):
    _test_resume_random_dataloader_from_iter("cpu", setup_sampler_fn)
    _test_resume_random_dataloader_from_iter("cpu", setup_sampler_fn, sampler_type="weighted")
    _test_resume_random_dataloader_from_iter("cpu", setup_sampler_fn, sampler_type="distributed")


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

        for resume_epoch in range(2, max_epochs):

            seen_batchs = []

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = Engine(update_fn)

            @engine.on(Events.EPOCH_STARTED(once=resume_epoch))
            def sync_seed(e):
                e.sync_dataflow()

            torch.manual_seed(12)
            engine.run(
                infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length,
            )

            batch_checker = BatchChecker(seen_batchs, init_counter=(resume_epoch - 1) * epoch_length)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_epoch, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)
            resume_state_dict = dict(epoch=resume_epoch, max_epochs=max_epochs, epoch_length=epoch_length)
            engine.load_state_dict(resume_state_dict)
            torch.manual_seed(12)
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

        for resume_iteration in range(2, min(num_iters * max_epochs, epoch_length * max_epochs), 7):

            seen_batchs = []

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = Engine(update_fn)

            @engine.on(Events.GET_BATCH_STARTED(once=resume_iteration - 1))
            def sync_seed(e):
                e.sync_dataflow()

            torch.manual_seed(12)
            engine.run(
                infinite_data_iterator(), max_epochs=max_epochs, epoch_length=epoch_length,
            )

            batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration - 1)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), "{} | {}: {} vs {}".format(
                    resume_iteration, batch_checker.counter, batch_checker.true_batch, batch
                )

            engine = Engine(update_fn)
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
def test_distrib_gpu(distributed_context_single_node_nccl, setup_sampler_fn):
    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler_fn)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler_fn)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo, setup_sampler_fn):
    device = "cpu"
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler_fn)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler_fn)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo, setup_sampler_fn):
    device = "cpu"
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler_fn)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler_fn)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl, setup_sampler_fn):
    device = "cuda:{}".format(distributed_context_multi_node_nccl["local_rank"])
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device, setup_sampler_fn)
    _test_resume_random_dataloader_from_epoch(device, setup_sampler_fn)
