import torch

import pytest


class IterationCounter(object):
    def __init__(self, start_value=1):
        self.current_iteration_count = start_value

    def __call__(self, engine):
        assert engine.state.iteration == self.current_iteration_count
        self.current_iteration_count += 1


class EpochCounter(object):
    def __init__(self, start_value):
        self.current_epoch_count = start_value

    def __call__(self, engine):
        assert engine.state.epoch == self.current_epoch_count
        self.current_epoch_count += 1


@pytest.fixture()
def counter_factory():
    def create(name, start_value=1):
        if name == "epoch":
            return EpochCounter(start_value)
        elif name == "iter":
            return IterationCounter(start_value)
        else:
            raise RuntimeError()

    return create


@pytest.fixture()
def setup_sampler_fn():
    def _setup_sampler(sampler_type, num_iters, batch_size):
        if sampler_type is None:
            return None

        if sampler_type == "weighted":
            from torch.utils.data.sampler import WeightedRandomSampler

            w = torch.ones(num_iters * batch_size, dtype=torch.float)
            for i in range(num_iters):
                w[batch_size * i : batch_size * (i + 1)] += i * 1.0
            return WeightedRandomSampler(w, num_samples=num_iters * batch_size, replacement=True)

        if sampler_type == "distributed":
            from torch.utils.data.distributed import DistributedSampler
            import torch.distributed as dist

            num_replicas = 1
            rank = 0
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()

            dataset = torch.zeros(num_iters * batch_size)
            return DistributedSampler(dataset, num_replicas=num_replicas, rank=rank)

    return _setup_sampler
