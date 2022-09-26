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


def _test_distrib_config(local_rank, backend, ws, true_device, rank=None, true_init_method=None):
    assert idist.backend() == backend, f"{idist.backend()} vs {backend}"

    this_device = idist.device()
    assert isinstance(this_device, torch.device)
    if backend in ("nccl", "gloo", "horovod") and "cuda" in this_device.type:
        assert this_device.type == torch.device(true_device).type, f"{this_device} vs {true_device}"
    elif backend in ("gloo", "horovod"):
        assert this_device.type == torch.device(true_device).type
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

    if idist.model_name() == "native-dist":
        from ignite.distributed.utils import _model

        if true_init_method is not None:
            assert _model._init_method == true_init_method


def _test_sync(cls):
    from ignite.distributed.utils import _SerialModel, _set_model

    _set_model(_SerialModel())

    sync()

    from ignite.distributed.utils import _model

    assert isinstance(_model, cls), f"{type(_model)} vs {cls}"


def _test_distrib__get_max_length(device):
    ws = idist.get_world_size()
    x = "_test_distrib__get_max_length" * (idist.get_rank() + 2)

    from ignite.distributed.utils import _model

    res = _model._get_max_length(x, device)
    assert res == len("_test_distrib__get_max_length" * (ws + 1))


def _test_distrib_all_reduce(device):

    res = idist.all_reduce(10)
    assert res == 10 * idist.get_world_size()

    t = torch.tensor(10, device=device)
    res = idist.all_reduce(t)
    assert res.item() == 10 * idist.get_world_size()

    rank = idist.get_rank()
    t = torch.tensor(rank * 2.0 + 1.0, device=device)
    res = idist.all_reduce(t)
    assert res.item() == sum([i * 2.0 + 1.0 for i in range(idist.get_world_size())])

    t = torch.tensor(rank * 2.0 + 1.0, device=device)
    res = idist.all_reduce(t, "MIN").item()
    true_val = min([i * 2 + 1 for i in range(idist.get_world_size())])
    assert res == true_val, f"{res} vs {true_val}"

    t = torch.tensor(rank * 2.0 + 1.0, device=device)
    res = idist.all_reduce(t, "MAX").item()
    true_val = max([i * 2.0 + 1.0 for i in range(idist.get_world_size())])
    assert res == true_val, f"{res} vs {true_val}"

    t = torch.ones(4, 4, device=device) * (rank * 2.0 + 1.0)
    res = idist.all_reduce(t, "MAX")
    true_val = torch.ones(4, 4, device=device) * ((idist.get_world_size() - 1) * 2.0 + 1.0)
    assert res.equal(true_val), f"{res} vs {true_val}"

    t = torch.tensor(rank * 2.0 + 1.0, device=device)
    res = idist.all_reduce(t, "PRODUCT").item()
    true_val = 1
    for v in [i * 2.0 + 1.0 for i in range(idist.get_world_size())]:
        true_val *= v
    assert res == true_val, f"{res} vs {true_val}"

    if idist.get_world_size() > 1:
        with pytest.raises(TypeError, match=r"Unhandled input type"):
            idist.all_reduce("abc")

        with pytest.raises(ValueError, match=r"Unsupported reduction operation"):
            idist.all_reduce(10, op="ABC")

        t = torch.tensor([0, 1, 2])
        res = idist.all_reduce(t)
        assert res.device == t.device, f"{res.device} vs {t.device}"


def _test_distrib_all_gather(device):

    res = torch.tensor(idist.all_gather(10), device=device)
    true_res = torch.tensor([10] * idist.get_world_size(), device=device)
    assert (res == true_res).all()

    t = torch.tensor(idist.get_rank(), device=device)
    res = idist.all_gather(t)
    true_res = torch.tensor([i for i in range(idist.get_world_size())], device=device)
    assert (res == true_res).all()

    x = "test-test"
    if idist.get_rank() == 0:
        x = "abc"
    res = idist.all_gather(x)
    true_res = ["abc"] + ["test-test"] * (idist.get_world_size() - 1)
    assert res == true_res

    base_x = "tests/ignite/distributed/utils/test_native.py" * 2000
    x = base_x
    if idist.get_rank() == 0:
        x = "abc"

    res = idist.all_gather(x)
    true_res = ["abc"] + [base_x] * (idist.get_world_size() - 1)
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


def _test_distrib_all_gather_group(device):

    if idist.get_world_size() > 1:
        ranks = [0, 1]
        rank = idist.get_rank()

        t = torch.tensor([rank], device=idist.device())
        group = idist.new_group(ranks)
        res = idist.all_gather(t, group=group)
        assert torch.equal(res, torch.tensor(ranks))

        t = torch.tensor([rank], device=device)
        res = idist.all_gather(t, group=ranks)
        assert torch.equal(res, torch.tensor(ranks))

        ranks = "abc"
        with pytest.raises(ValueError, match=r"Argument ranks should be list of int"):
            group = idist.new_group(ranks)
            res = idist.all_gather(t, group=group)


def _test_distrib_broadcast(device):

    rank = idist.get_rank()
    ws = idist.get_world_size()

    def _test(data_src, data_others, safe_mode):
        for src in range(ws):

            data = data_src if rank == src else data_others
            res = idist.broadcast(data, src=src, safe_mode=safe_mode)

            if isinstance(res, torch.Tensor):
                assert (res == data_src).all(), f"{res} vs {data_src}"
                assert data_src.dtype == res.dtype
            else:
                assert res == data_src, f"{res} vs {data_src}"

    _test(10, 0, safe_mode=False)
    _test(10, None, safe_mode=True)

    t = torch.tensor([1.2345, 2.3456], dtype=torch.float, device=device)
    _test(t, torch.empty_like(t), safe_mode=False)
    _test(t, None, safe_mode=True)
    _test(t, "abc", safe_mode=True)

    _test("test-abcdefg", "", safe_mode=False)
    _test("test-abcdefg", None, safe_mode=True)
    _test("test-abcdefg", 1.2, safe_mode=True)

    s = "tests/ignite/distributed/utils/test_horovod.py::test_idist_broadcast_hvd" * 200
    _test(s, "", safe_mode=False)
    _test(s, None, safe_mode=True)
    _test(s, 123.0, safe_mode=True)

    t = torch.arange(100, device=device).reshape(4, 25) * 2
    _test(t, torch.empty_like(t), safe_mode=False)
    _test(t, None, safe_mode=True)
    _test(t, "None", safe_mode=True)

    t = torch.tensor(12)
    _test(t, torch.empty_like(t), safe_mode=False)
    _test(t, None, safe_mode=True)
    _test(t, 123.4, safe_mode=True)

    if idist.get_world_size() > 1:
        with pytest.raises(TypeError, match=r"Unhandled input type"):
            idist.broadcast([0, 1, 2], src=0)

    if idist.get_world_size() > 1:
        msg = "Source data can not be None" if rank == 0 else "Argument safe_mode should be True"
        with pytest.raises(ValueError, match=msg):
            idist.broadcast(None, src=0)


def _test_distrib_barrier(device):

    t = torch.tensor([idist.get_rank()], device=device, dtype=torch.float)
    true_res = sum([i for i in range(idist.get_world_size())])

    if idist.get_rank() == 0:
        t += 10.0
    idist.barrier()

    tt = idist.all_reduce(t)
    assert tt.item() == true_res + 10.0


def _test_distrib_new_group(device):

    if idist.get_world_size() > 1 and idist.backend() is not None:
        bnd = idist.backend()
        ranks = [0, 1]
        if idist.has_native_dist_support and bnd in ("nccl", "gloo", "mpi"):

            g1 = idist.new_group(ranks)
            g2 = dist.new_group(ranks)

            rank = idist.get_rank()
            if rank in ranks:
                assert g1.rank() == g2.rank()
        elif idist.has_xla_support and bnd in ("xla-tpu"):

            assert idist.new_group(ranks) == [ranks]
        elif idist.has_hvd_support and bnd in ("horovod"):
            from horovod.common.process_sets import ProcessSet

            g1 = idist.new_group(ranks)
            g2 = ProcessSet(ranks)

            rank = idist.get_rank()
            if rank in ranks:
                assert g1.ranks == g2.ranks

    elif idist.backend() is None:
        ranks = [0, 1]
        assert idist.new_group(ranks) == ranks

    with pytest.raises(ValueError, match="Argument ranks should be list of int"):
        ranks = ["a", "b", "c"]
        idist.new_group(ranks)

    with pytest.raises(ValueError, match="Argument ranks should be list of int"):
        ranks = 1
        idist.new_group(ranks)


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
