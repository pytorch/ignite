import random
import numpy as np
import torch

from ignite.engine import Engine
from ignite.engine.utils import ReproducibleBatchSampler, _update_dataloader
from ignite.engine.utils import keep_random_state


def test__update_dataloader(setup_sampler_fn):
    def _test(sampler_type=None):
        num_epochs = 3
        batch_size = 4
        num_iters = 17
        data = torch.randint(0, 1000, size=(num_iters * batch_size,))
        num_workers = 4

        sampler = setup_sampler_fn(sampler_type, num_iters, batch_size)
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

        sampler = setup_sampler_fn(sampler_type, num_iters, batch_size)
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
        new_dataloader = _update_dataloader(dataloader, ReproducibleBatchSampler(batch_sampler))

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


def test_keep_random_state():

    Engine._manual_seed(54, 0)
    true_values = []
    for _ in range(5):
        true_values.append([torch.tensor([random.random()]), torch.rand(2), torch.from_numpy(np.random.rand(2))])

    @keep_random_state
    def user_handler():
        Engine._manual_seed(22, 0)
        _ = [random.random(), torch.rand(2), np.random.rand(2)]

    Engine._manual_seed(54, 0)
    res_values = []
    for _ in range(5):
        res_values.append([torch.tensor([random.random()]), torch.rand(2), torch.from_numpy(np.random.rand(2))])
        user_handler()

    for a, b in zip(true_values, res_values):
        for i, j in zip(a, b):
            assert (i == j).all()
