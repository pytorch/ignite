import pytest
import torch
import torch.distributed as dist

import ignite.distributed as idist
from ignite.distributed.utils import _rank_not_in_group, all_gather_tensors_with_shapes, sync
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


def _test_distrib_all_reduce_group(device):
    assert idist.get_world_size() > 1, idist.get_world_size()
    assert idist.backend() is not None, idist.backend()

    ranks = sorted(range(idist.get_world_size() - 1, 0, -1))  # [0, 1, 2, 3] -> [3, 2, 1]
    rank = idist.get_rank()
    t = torch.tensor([rank], device=device)
    bnd = idist.backend()

    group = idist.new_group(ranks)
    if bnd in ("horovod"):
        with pytest.raises(NotImplementedError, match=r"all_reduce with group for horovod is not implemented"):
            res = idist.all_reduce(t, group=group)
    else:
        if rank in ranks:
            # we should call all_reduce with group on the participating ranks only
            # otherwise a warning is raised:
            # UserWarning: Running all_reduce on global rank 2 which does not belong to the given group.
            res = idist.all_reduce(t, group=group)
            assert res == torch.tensor([sum(ranks)], device=device)

    t = torch.tensor([rank], device=device)
    if bnd in ("horovod"):
        with pytest.raises(NotImplementedError, match=r"all_reduce with group for horovod is not implemented"):
            res = idist.all_reduce(t, group=ranks)
    else:
        if rank in ranks:
            res = idist.all_reduce(t, group=ranks)
            assert res == torch.tensor([sum(ranks)], device=device)

    if bnd in ("nccl", "gloo", "mpi"):
        with pytest.raises(ValueError, match=r"Argument group should be list of int or ProcessGroup"):
            idist.all_reduce(t, group="abc")
    elif bnd in ("xla-tpu"):
        with pytest.raises(ValueError, match=r"Argument group should be list of int"):
            idist.all_reduce(t, group="abc")
    elif bnd in ("horovod"):
        with pytest.raises(NotImplementedError, match=r"all_reduce with group for horovod is not implemented"):
            idist.all_reduce(t, group="abc")


def _test_distrib_all_gather(device):
    rank = idist.get_rank()
    ws = idist.get_world_size()

    res = torch.tensor(idist.all_gather(10), device=device)
    true_res = torch.tensor([10] * ws, device=device)
    assert (res == true_res).all()

    t = torch.tensor(rank, device=device)
    res = idist.all_gather(t)
    true_res = torch.tensor([i for i in range(ws)], device=device)
    assert (res == true_res).all()

    x = "test-test"
    if rank == 0:
        x = "abc"
    res = idist.all_gather(x)
    true_res = ["abc"] + ["test-test"] * (ws - 1)
    assert res == true_res

    base_x = "tests/ignite/distributed/utils/test_native.py" * 2000
    x = base_x
    if rank == 0:
        x = "abc"

    res = idist.all_gather(x)
    true_res = ["abc"] + [base_x] * (ws - 1)
    assert res == true_res

    t = torch.arange(100, device=device).reshape(4, 25) * (rank + 1)
    in_dtype = t.dtype
    res = idist.all_gather(t)
    assert res.shape == (ws * 4, 25)
    assert res.dtype == in_dtype
    true_res = torch.zeros(ws * 4, 25, device=device)
    for i in range(ws):
        true_res[i * 4 : (i + 1) * 4, ...] = torch.arange(100, device=device).reshape(4, 25) * (i + 1)
    assert (res == true_res).all()

    if ws > 1 and idist.backend() != "xla-tpu":
        t = {
            "a": [rank + 1, rank + 2, torch.tensor(rank + 3, device=device)],
            "b": torch.tensor([[rank + 1, rank + 2, rank + 3]], device=device),
            "c": {"abcd": rank, "cdfg": torch.tensor(rank, dtype=torch.uint8, device=device)},
        }
        res = idist.all_gather(t)
        assert isinstance(res, list) and len(res) == ws
        for i, obj in enumerate(res):
            assert isinstance(obj, dict)
            assert list(obj.keys()) == ["a", "b", "c"], obj
            expected_device = (
                device if torch.device(device).type == "cpu" else torch.device(f"{torch.device(device).type}:{i}")
            )
            expected = {
                "a": [i + 1, i + 2, torch.tensor(i + 3, device=expected_device)],
                "b": torch.tensor([[i + 1, i + 2, i + 3]], device=expected_device),
                "c": {"abcd": i, "cdfg": torch.tensor(i, dtype=torch.uint8, device=expected_device)},
            }
            assert obj["a"] == expected["a"]
            assert (obj["b"] == expected["b"]).all()
            assert obj["c"] == expected["c"]


def _test_distrib_all_gather_group(device):
    assert idist.get_world_size() > 1, idist.get_world_size()

    ranks = sorted(range(idist.get_world_size() - 1, 0, -1))  # [0, 1, 2, 3] -> [3, 2, 1]
    rank = idist.get_rank()
    bnd = idist.backend()

    t = torch.tensor([rank], device=device)
    group = idist.new_group(ranks)
    res = idist.all_gather(t, group=group)
    if rank in ranks:
        assert torch.equal(res, torch.tensor(ranks, device=device))
    else:
        assert res == t

    t = torch.tensor([rank], device=device)
    if bnd == "horovod":
        res = idist.all_gather(t, group=group)
    else:
        res = idist.all_gather(t, group=ranks)
    if rank in ranks:
        assert torch.equal(res, torch.tensor(ranks, device=device))
    else:
        assert res == t

    t = {
        "a": [rank + 1, rank + 2, torch.tensor(rank + 3, device=device)],
        "b": torch.tensor([[rank + 1, rank + 2, rank + 3]], device=device),
        "c": {"abcd": rank, "cdfg": torch.tensor(rank, dtype=torch.uint8, device=device)},
    }
    if bnd in ("xla-tpu"):
        with pytest.raises(NotImplementedError, match=r"all_gather on object is not implemented for xla"):
            res = idist.all_gather(t, group=ranks)
    elif bnd in ("horovod"):
        with pytest.raises(NotImplementedError, match=r"all_gather with group for horovod is not implemented"):
            res = idist.all_gather(t, group=group)
    else:
        res = idist.all_gather(t, group=ranks)
        if rank in ranks:
            assert isinstance(res, list) and len(res) == len(ranks)
            for i, obj in zip(ranks, res):
                assert isinstance(obj, dict)
                assert list(obj.keys()) == ["a", "b", "c"], obj
                expected_device = (
                    device if torch.device(device).type == "cpu" else torch.device(f"{torch.device(device).type}:{i}")
                )
                expected = {
                    "a": [i + 1, i + 2, torch.tensor(i + 3, device=expected_device)],
                    "b": torch.tensor([[i + 1, i + 2, i + 3]], device=expected_device),
                    "c": {"abcd": i, "cdfg": torch.tensor(i, dtype=torch.uint8, device=expected_device)},
                }
                assert obj["a"] == expected["a"], (obj, expected)
                assert (obj["b"] == expected["b"]).all(), (obj, expected)
                assert obj["c"] == expected["c"], (obj, expected)
        else:
            assert res == t

    t = torch.tensor([rank], device=device)
    if bnd in ("nccl", "gloo", "mpi", "horovod"):
        with pytest.raises(ValueError, match=r"Argument group should be list of int"):
            res = idist.all_gather(t, group="abc")
    elif bnd in ("xla-tpu"):
        with pytest.raises(ValueError, match=r"Argument group should be list of int"):
            res = idist.all_gather(t, group="abc")


def _test_idist_all_gather_tensors_with_shapes(device):
    torch.manual_seed(41)
    rank = idist.get_rank()
    ws = idist.get_world_size()
    reference = torch.randn(ws * 5, ws * 5, ws * 5, device=device)
    rank_tensor = reference[
        rank * (rank + 1) // 2 : rank * (rank + 1) // 2 + rank + 1,
        rank * (rank + 3) // 2 : rank * (rank + 3) // 2 + rank + 2,
        rank * (rank + 5) // 2 : rank * (rank + 5) // 2 + rank + 3,
    ]
    tensors = all_gather_tensors_with_shapes(rank_tensor, [[r + 1, r + 2, r + 3] for r in range(ws)])
    for r in range(ws):
        r_tensor = reference[
            r * (r + 1) // 2 : r * (r + 1) // 2 + r + 1,
            r * (r + 3) // 2 : r * (r + 3) // 2 + r + 2,
            r * (r + 5) // 2 : r * (r + 5) // 2 + r + 3,
        ]
        assert r_tensor.allclose(tensors[r])


def _test_idist_all_gather_tensors_with_shapes_group(device):
    assert idist.get_world_size(), idist.get_world_size()
    torch.manual_seed(41)

    rank = idist.get_rank()
    ranks = sorted(range(idist.get_world_size() - 1, 0, -1))  # [0, 1, 2, 3] -> [1, 2, 3]
    ws = idist.get_world_size()
    if rank in ranks:
        reference = torch.randn(ws * 5, ws * 5, ws * 5, device=device)
        rank_tensor = reference[
            rank * (rank + 1) // 2 : rank * (rank + 1) // 2 + rank + 1,
            rank * (rank + 3) // 2 : rank * (rank + 3) // 2 + rank + 2,
            rank * (rank + 5) // 2 : rank * (rank + 5) // 2 + rank + 3,
        ]
    else:
        rank_tensor = torch.tensor([rank], device=device)

    tensors = all_gather_tensors_with_shapes(rank_tensor, [[r + 1, r + 2, r + 3] for r in ranks], ranks)
    if rank in ranks:
        for i, r in enumerate(ranks):
            r_tensor = reference[
                r * (r + 1) // 2 : r * (r + 1) // 2 + r + 1,
                r * (r + 3) // 2 : r * (r + 3) // 2 + r + 2,
                r * (r + 5) // 2 : r * (r + 5) // 2 + r + 3,
            ]
            assert r_tensor.allclose(tensors[i])
    else:
        assert [rank_tensor] == tensors


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


def _test_distrib_group(device):
    ranks = sorted(range(idist.get_world_size() - 1, 0, -1))  # [0, 1, 2, 3] -> [1, 2, 3]
    if idist.get_world_size() > 1 and idist.backend() is not None:
        bnd = idist.backend()
        rank = idist.get_rank()
        g = idist.new_group(ranks)
        if idist.has_native_dist_support and bnd in ("nccl", "gloo", "mpi"):
            if rank in ranks:
                # mapping between group ranks and global ranks
                global_to_group = {r: i for i, r in enumerate(ranks)}
                assert g.rank() == global_to_group[rank], (g.rank(), global_to_group, rank)

        elif idist.has_xla_support and bnd in ("xla-tpu"):
            assert g == [ranks]
        elif idist.has_hvd_support and bnd in ("horovod"):
            if rank in ranks:
                assert g.ranks == ranks

        if rank in ranks:
            assert not _rank_not_in_group(g)
        else:
            assert _rank_not_in_group(g)

    elif idist.backend() is None:
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
            value.add_(torch.tensor(100).to(device))

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
