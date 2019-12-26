import os
import pytest

try:
    from collections.abc import Mapping
except ImportError:  # Python 2.7 compatibility
    from collections import Mapping

import torch

from ignite.engine import Engine, State, Events
from ignite.engine.engine import ReproducibleBatchSampler, _update_dataloader


def test_state_dict():
    engine = Engine(lambda e, b: 1)
    sd = engine.state_dict()
    assert isinstance(sd, Mapping) and len(sd) == 0

    def _test(state):
        engine.state = state
        sd = engine.state_dict()
        assert isinstance(sd, Mapping) and \
            len(sd) == len(engine._state_dict_all_req_keys) + 1
        assert sd['seed'] == engine.state.seed
        assert sd['iteration'] == engine.state.iteration
        assert sd['epoch_length'] == engine.state.epoch_length
        assert sd['max_epochs'] == engine.state.max_epochs

    _test(State(seed=0, iteration=500, epoch_length=1000, max_epochs=100))
    _test(State(seed=0, epoch=5, epoch_length=1000, max_epochs=100))


def test_state_dict_integration():
    engine = Engine(lambda e, b: 1)
    data = list(range(100))
    engine.run(data, max_epochs=10, seed=17)
    sd = engine.state_dict()
    assert isinstance(sd, Mapping) and len(sd) == 4
    assert sd['seed'] == engine.state.seed
    assert sd['iteration'] == engine.state.iteration == 10 * 100
    assert sd['epoch_length'] == engine.state.epoch_length == 100
    assert sd['max_epochs'] == engine.state.max_epochs == 10


def test_load_state_dict_asserts():
    engine = Engine(lambda e, b: 1)

    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary"):
        engine.load_state_dict("123")

    with pytest.raises(ValueError, match=r"is absent in provided state_dict"):
        engine.load_state_dict({})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        engine.load_state_dict({"seed": 0, "max_epochs": 100, "epoch_length": 120})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        engine.load_state_dict({"seed": 0, "max_epochs": 100, "epoch_length": 120,
                                "iteration": 12, "epoch": 123})


def test_load_state_dict():
    engine = Engine(lambda e, b: 1)

    def _test(sd):
        engine.load_state_dict(sd)
        assert sd['seed'] == engine.state.seed
        if 'iteration' in sd:
            assert sd['iteration'] == engine.state.iteration
        elif 'epoch' in sd:
            assert sd['epoch'] == engine.state.epoch
        assert sd['epoch_length'] == engine.state.epoch_length
        assert sd['max_epochs'] == engine.state.max_epochs

    _test({"seed": 0, "max_epochs": 100, "epoch_length": 120, "iteration": 123})
    _test({"seed": 0, "max_epochs": 100, "epoch_length": 120, "epoch": 5})


def test_load_state_dict_integration(counter_factory):
    engine = Engine(lambda e, b: 1)

    state_dict = {"seed": 0, "max_epochs": 100, "epoch_length": 120, "epoch": 5}

    engine.load_state_dict(state_dict)
    engine.add_event_handler(Events.ITERATION_COMPLETED, counter_factory('iter', 5 * 120 + 1))
    engine.add_event_handler(Events.EPOCH_COMPLETED, counter_factory('epoch', 6))
    data = list(range(120))
    engine.run(data)


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

        def update_fn(engine, batch):
            assert batch_checker.check(batch), \
                "{}: {} vs {}".format(batch_checker.counter, batch_checker.true_batch, batch)

        engine = Engine(update_fn)
        engine.run(data, max_epochs=max_epochs, epoch_length=num_iters)
        if num_iters is None:
            num_iters = len(data)
        assert engine.state.iteration == num_iters * max_epochs
        assert engine.state.epoch == max_epochs

    def _test_as_iter(data, max_epochs, num_iters):

        batch_checker = BatchChecker(data)

        def update_fn(engine, batch):
            assert batch_checker.check(batch), \
                "{}: {} vs {}".format(batch_checker.counter, batch_checker.true_batch, batch)

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


def test_strict_resume_from_iter():

    def _test(epoch_length=None):

        max_epochs = 5
        num_iters = 21
        data = torch.randint(0, 1000, size=(num_iters,))
        if epoch_length is None:
            epoch_length = num_iters

        for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs), 4):
            batch_checker = BatchChecker(data, init_counter=resume_iteration)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), \
                    "{} | {}: {} vs {}".format(
                        resume_iteration,
                        batch_checker.counter, batch_checker.true_batch, batch)

            engine = Engine(update_fn)

            @engine.on(Events.EPOCH_COMPLETED)
            def check_iteration(engine):
                assert engine.state.iteration == batch_checker.counter

            resume_state_dict = {
                "iteration": resume_iteration,
                "max_epochs": max_epochs,
                "epoch_length": epoch_length,
                "seed": 0
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

        for resume_epoch in range(1, max_epochs):
            batch_checker = BatchChecker(data, init_counter=resume_epoch * epoch_length)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), \
                    "{} | {}: {} vs {}".format(
                        resume_epoch,
                        batch_checker.counter, batch_checker.true_batch, batch)

            engine = Engine(update_fn)
            resume_state_dict = dict(epoch=resume_epoch,
                                     max_epochs=max_epochs,
                                     epoch_length=epoch_length,
                                     seed=0)
            engine.load_state_dict(resume_state_dict)
            engine.run(data)
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(60)
    _test(15)


def test_resume_dataloader_from_iter():

    def _test(epoch_length=None):
        max_epochs = 3
        batch_size = 4
        num_iters = 11
        data = torch.randint(0, 1000, size=(num_iters * batch_size, 2, 4, 5))
        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                                                     drop_last=False, shuffle=False)

            for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs), 6):
                batch_checker = BatchChecker(data.reshape(-1, batch_size, 2, 4, 5), init_counter=resume_iteration)

                def update_fn(engine, batch):
                    assert batch_checker.check(batch), \
                        "{} {} | {}: {} vs {}".format(
                            num_workers, resume_iteration,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(iteration=resume_iteration,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=0)
                engine.load_state_dict(resume_state_dict)
                engine.run(dataloader)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs, \
                    "{}, {} | {} vs {}".format(num_workers, resume_iteration,
                                               engine.state.iteration,
                                               epoch_length * max_epochs)

    _test()
    _test(epoch_length=30)
    _test(epoch_length=5)


def test_resume_dataloader_from_epoch():

    def _test(epoch_length=None):

        max_epochs = 5
        batch_size = 4
        num_iters = 11
        data = torch.randint(0, 256, size=(num_iters * batch_size, 3, 4, 5))
        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers,
                                                     drop_last=False, shuffle=False)

            for resume_epoch in range(1, max_epochs):
                batch_checker = BatchChecker(data.reshape(-1, batch_size, 3, 4, 5),
                                             init_counter=resume_epoch * epoch_length)

                def update_fn(engine, batch):
                    assert batch_checker.check(batch), \
                        "{} {} | {}: {} vs {}".format(
                            num_workers, resume_epoch,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(epoch=resume_epoch,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=0)
                engine.load_state_dict(resume_state_dict)
                engine.run(dataloader)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(epoch_length=60)
    _test(epoch_length=15)


def test_reproduce_run_with_seed():

    def _test(epoch_length=None):
        max_epochs = 5
        batch_size = 4
        num_iters = 21
        data = torch.randint(0, 1000, size=(num_iters * batch_size, ))

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     drop_last=True, shuffle=True)

            ref_seen_batchs = []

            def ref_update_fn(engine, batch):
                ref_seen_batchs.append(batch)

            engine = Engine(ref_update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            seen_batchs = []

            def update_fn(engine, batch):
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            for i, (ref_b, b) in enumerate(zip(ref_seen_batchs, seen_batchs)):
                assert (ref_b == b).all(), "{}, {}: {} vs {}".format(num_workers, i, ref_b, b)

    _test()
    _test(60)
    _test(15)


def _test_resume_random_dataloader_from_epoch(device):

    torch.manual_seed(0)

    def _test(epoch_length=None):
        max_epochs = 5
        batch_size = 4
        num_iters = 21
        data = torch.randint(0, 1000, size=(num_iters * batch_size, ))

        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory="cuda" in device,
                                                     drop_last=True, shuffle=True)

            seen_batchs = []

            def update_fn(engine, batch):
                batch_to_device = batch.to(device)
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            for resume_epoch in range(1, max_epochs):
                batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

                def update_fn(engine, batch):
                    batch_to_device = batch.to(device)
                    assert batch_checker.check(batch), \
                        "{} {} | {}: {} vs {}".format(
                            num_workers, resume_epoch,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(epoch=resume_epoch,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=12)
                engine.load_state_dict(resume_state_dict)
                engine.run(dataloader)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs

    _test()
    _test(60)
    _test(15)


def test_resume_random_dataloader_from_epoch():
    _test_resume_random_dataloader_from_epoch("cpu")


def _test_resume_random_dataloader_from_iter(device):

    torch.manual_seed(0)

    def _test(epoch_length=None):
        max_epochs = 3
        batch_size = 4
        num_iters = 17
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))

        if epoch_length is None:
            epoch_length = num_iters

        for num_workers in [0, 4]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory="cuda" in device,
                                                     drop_last=True, shuffle=True)
            seen_batchs = []

            def update_fn(engine, batch):
                batch_to_device = batch.to(device)
                seen_batchs.append(batch)

            engine = Engine(update_fn)
            engine.run(dataloader, max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

            for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs), 7):
                batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration)

                def update_fn(engine, batch):
                    batch_to_device = batch.to(device)
                    assert batch_checker.check(batch), \
                        "{} {} | {}: {} vs {}".format(
                            num_workers, resume_iteration,
                            batch_checker.counter, batch_checker.true_batch, batch)

                engine = Engine(update_fn)
                resume_state_dict = dict(iteration=resume_iteration,
                                         max_epochs=max_epochs,
                                         epoch_length=epoch_length,
                                         seed=12)
                engine.load_state_dict(resume_state_dict)
                engine.run(dataloader)
                assert engine.state.epoch == max_epochs
                assert engine.state.iteration == epoch_length * max_epochs, \
                    "{}, {} | {} vs {}".format(num_workers, resume_iteration,
                                               engine.state.iteration,
                                               epoch_length * max_epochs)

    _test()
    _test(50)
    _test(11)


def test_resume_random_dataloader_from_iter():
    _test_resume_random_dataloader_from_iter("cpu")


def test_reproducible_batch_sampler():
    import torch
    from torch.utils.data import DataLoader

    data = list(range(100))
    dataloader = DataLoader(data, batch_size=12, num_workers=0, shuffle=True, drop_last=True)

    torch.manual_seed(12 + 0)
    dataloader_ = _update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))

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
        dataloader_ = _update_dataloader(dataloader, ReproducibleBatchSampler(dataloader.batch_sampler))
        resumed_seen_batches = []
        for b in dataloader_:
            resumed_seen_batches.append(b)

        assert all([(b1 == b2).all() for b1, b2 in zip(seen_batches[resume_epoch], resumed_seen_batches)])


def _test_resume_random_data_iterator_from_epoch(device):

    def _test(epoch_length=None):
        max_epochs = 5
        batch_size = 4
        num_iters = 21

        def infinite_data_iterator():
            torch.manual_seed(0)
            while True:
                for _ in range(num_iters):
                    data = torch.randint(0, 1000, size=(batch_size,), device=device)
                    yield data

        if epoch_length is None:
            epoch_length = num_iters

        seen_batchs = []

        def update_fn(engine, batch):
            seen_batchs.append(batch)

        engine = Engine(update_fn)
        engine.run(infinite_data_iterator(), max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

        for resume_epoch in range(1, max_epochs):
            batch_checker = BatchChecker(seen_batchs, init_counter=resume_epoch * epoch_length)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), \
                    "{} | {}: {} vs {}".format(resume_epoch, batch_checker.counter, batch_checker.true_batch, batch)

            engine = Engine(update_fn)
            resume_state_dict = dict(epoch=resume_epoch,
                                     max_epochs=max_epochs,
                                     epoch_length=epoch_length,
                                     seed=12)
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
            torch.manual_seed(0)
            while True:
                for _ in range(num_iters):
                    data = torch.randint(0, 1000, size=(batch_size,), device=device)
                    yield data

        if epoch_length is None:
            epoch_length = num_iters

        seen_batchs = []

        def update_fn(engine, batch):
            seen_batchs.append(batch)

        engine = Engine(update_fn)
        engine.run(infinite_data_iterator(), max_epochs=max_epochs, seed=12, epoch_length=epoch_length)

        for resume_iteration in range(1, min(num_iters * max_epochs, epoch_length * max_epochs), 7):
            batch_checker = BatchChecker(seen_batchs, init_counter=resume_iteration)

            def update_fn(engine, batch):
                assert batch_checker.check(batch), \
                    "{} | {}: {} vs {}".format(resume_iteration, batch_checker.counter, batch_checker.true_batch, batch)

            engine = Engine(update_fn)
            resume_state_dict = dict(iteration=resume_iteration,
                                     max_epochs=max_epochs,
                                     epoch_length=epoch_length,
                                     seed=12)
            engine.load_state_dict(resume_state_dict)
            engine.run(infinite_data_iterator())
            assert engine.state.epoch == max_epochs
            assert engine.state.iteration == epoch_length * max_epochs, \
                "{} | {} vs {}".format(resume_iteration, engine.state.iteration, epoch_length * max_epochs)

    _test()
    _test(50)
    _test(11)


def test_resume_random_data_iterator_from_iter():
    _test_resume_random_data_iterator_from_iter("cpu")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl):
    device = "cuda:{}".format(distributed_context_single_node_nccl['local_rank'])
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device)
    _test_resume_random_dataloader_from_epoch(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = "cpu"
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device)
    _test_resume_random_dataloader_from_epoch(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device)
    _test_resume_random_dataloader_from_epoch(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _test_resume_random_data_iterator_from_iter(device)
    _test_resume_random_data_iterator_from_epoch(device)
    _test_resume_random_dataloader_from_iter(device)
    _test_resume_random_dataloader_from_epoch(device)
