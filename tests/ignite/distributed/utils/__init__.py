import pytest
import torch
import torch.distributed as dist

import ignite.distributed as idist
from ignite.distributed.utils import sync
from ignite.engine import Engine, Events


def _sanity_check():
    from ignite.distributed.utils import _model

    assert _model.get_world_size() == _model.get_nnodes() * _model.get_nproc_per_node()
    assert _model.get_local_rank() < _model.get_nproc_per_node()
    assert _model.get_rank() < _model.get_world_size()
    assert _model.get_node_rank() < _model.get_nnodes()


def _test_distrib_config(local_rank, backend, ws, true_device, rank=None):
    assert idist.backend() == backend, "{} vs {}".format(idist.backend(), backend)

    this_device = idist.device()
    assert isinstance(this_device, torch.device)
    if backend in ("nccl", "horovod") and "cuda" in this_device.type:
        true_device = torch.device("{}:{}".format(true_device, local_rank))
        assert this_device == true_device, "{} vs {}".format(this_device, true_device)
    elif backend in ("gloo", "horovod"):
        assert this_device == torch.device(true_device)
    elif backend == "xla-tpu":
        assert true_device in this_device.type

    if rank is None:
        if idist.model_name() == "native-dist":
            rank = dist.get_rank()

    if rank is not None:
        assert idist.get_rank() == rank

    assert idist.get_world_size() == ws
    assert idist.get_local_rank() == local_rank

    assert idist.model_name() in ("native-dist", "xla-dist", "horovod-dist")

    _sanity_check()


def _test_sync(cls):
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    sync()

    from ignite.distributed.utils import _model

    assert isinstance(_model, cls), "{} vs {}".format(type(_model), cls)


def _test_distrib_all_reduce(device):

    res = idist.all_reduce(10)
    assert res == 10 * idist.get_world_size()

    t = torch.tensor(10, device=device)
    res = idist.all_reduce(t)
    assert res.item() == 10 * idist.get_world_size()

    t = torch.tensor(idist.get_rank(), device=device)
    res = idist.all_reduce(t)
    assert res.item() == sum([i for i in range(idist.get_world_size())])

    if idist.get_world_size() > 1:
        with pytest.raises(TypeError, match=r"Unhandled input type"):
            idist.all_reduce("abc")

        with pytest.raises(ValueError, match=r"Unsupported reduction operation"):
            idist.all_reduce(10, op="ABC")


def _test_distrib_all_gather(device):

    res = idist.all_gather(10)
    true_res = torch.tensor([10,] * idist.get_world_size(), device=device)
    assert (res == true_res).all()

    t = torch.tensor(idist.get_rank(), device=device)
    res = idist.all_gather(t)
    true_res = torch.tensor([i for i in range(idist.get_world_size())], device=device)
    assert (res == true_res).all()

    x = "test-test"
    if idist.get_rank() == 0:
        x = "abc"
    res = idist.all_gather(x)
    true_res = ["abc",] + ["test-test"] * (idist.get_world_size() - 1)
    assert res == true_res

    base_x = "x" * 1026
    x = base_x
    if idist.get_rank() == 0:
        x = "abc"

    if idist.get_rank() > 0:
        with pytest.warns(UserWarning, match=r"is larger than 1024 and thus will be truncated"):
            res = idist.all_gather(x)
    else:
        res = idist.all_gather(x)
    true_res = ["abc",] + [base_x[:1024]] * (idist.get_world_size() - 1)
    assert res == true_res

    t = torch.arange(100, device=device).reshape(4, 25) * (idist.get_rank() + 1)
    in_dtype = t.dtype
    res = idist.all_gather(t)
    assert res.shape == (idist.get_world_size() * 4, 25)
    assert res.dtype == in_dtype
    true_res = torch.zeros(idist.get_world_size() * 4, 25, device=device)
    for i in range(idist.get_world_size()):
        true_res[i * 4 : (i + 1) * 4, ...] = torch.arange(100, device=device).reshape(4, 25) * (i + 1)
    assert (res == true_res).all()

    if idist.get_world_size() > 1:
        with pytest.raises(TypeError, match=r"Unhandled input type"):
            idist.all_reduce([0, 1, 2])


def _test_distrib_barrier(device):

    t = torch.tensor([idist.get_rank()], device=device, dtype=torch.float)
    true_res = sum([i for i in range(idist.get_world_size())])

    if idist.get_rank() == 0:
        t += 10.0
    idist.barrier()

    tt = idist.all_reduce(t)
    assert tt.item() == true_res + 10.0


def _test_distrib_one_rank_only(device):
    def _test(barrier):
        # last rank
        rank = idist.get_world_size() - 1

        value = torch.tensor(0).to(device)

        @idist.one_rank_only(rank=rank, with_barrier=barrier)
        def initialize():
            value.data = torch.tensor(100).to(device)

        initialize()

        value_list = idist.all_gather(tensor=value)

        for r in range(idist.get_world_size()):
            if r == rank:
                assert value_list[r].item() == 100
            else:
                assert value_list[r].item() == 0

    _test(barrier=True)
    _test(barrier=False)


def _test_distrib_one_rank_only_with_engine(device):
    def _test(barrier):
        engine = Engine(lambda e, b: b)

        batch_sum = torch.tensor(0).to(device)

        @engine.on(Events.ITERATION_COMPLETED)
        @idist.one_rank_only(with_barrier=barrier)  # ie rank == 0
        def _(_):
            batch_sum.data += torch.tensor(engine.state.batch).to(device)

        engine.run([1, 2, 3], max_epochs=2)

        value_list = idist.all_gather(tensor=batch_sum)

        for r in range(idist.get_world_size()):
            if r == 0:
                assert value_list[r].item() == 12
            else:
                assert value_list[r].item() == 0

    _test(barrier=True)
    _test(barrier=False)
