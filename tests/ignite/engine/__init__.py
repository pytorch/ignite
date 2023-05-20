import torch

try:
    from torch.utils.data import IterableDataset
except ImportError:

    class IterableDataset:
        pass


class BatchChecker:
    def __init__(self, data, init_counter=0):
        self.counter = init_counter
        self.data = data
        self.true_batch = None

    def check(self, batch):
        self.true_batch = self.data[self.counter % len(self.data)]
        self.counter += 1
        res = self.true_batch == batch
        return res.all() if not isinstance(res, bool) else res


class IterationCounter:
    def __init__(self, start_value=1):
        self.current_iteration_count = start_value

    def __call__(self, engine):
        assert engine.state.iteration == self.current_iteration_count
        self.current_iteration_count += 1


class EpochCounter:
    def __init__(self, start_value=1):
        self.current_epoch_count = start_value

    def __call__(self, engine):
        assert engine.state.epoch == self.current_epoch_count
        self.current_epoch_count += 1


def setup_sampler(sampler_type, num_iters, batch_size):
    if sampler_type is None:
        return None, batch_size

    if sampler_type == "weighted":
        from torch.utils.data.sampler import WeightedRandomSampler

        w = torch.ones(num_iters * batch_size, dtype=torch.float)
        for i in range(num_iters):
            w[batch_size * i : batch_size * (i + 1)] += i * 1.0
        return WeightedRandomSampler(w, num_samples=num_iters * batch_size, replacement=True), batch_size

    if sampler_type == "distributed":
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler

        num_replicas = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

        dataset = torch.zeros(num_iters * batch_size)
        return DistributedSampler(dataset, num_replicas=num_replicas, rank=rank), batch_size // num_replicas


class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))


def get_iterable_dataset(*args, **kwargs):
    return MyIterableDataset(*args, **kwargs)
