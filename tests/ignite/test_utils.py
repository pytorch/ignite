import os
import logging
import pytest
import torch
import torch.distributed as dist

from collections import namedtuple
from ignite.utils import convert_tensor, to_onehot, setup_logger, one_rank_only
from ignite.engine import Engine, Events


def test_convert_tensor():
    x = torch.tensor([0.0])
    tensor = convert_tensor(x)
    assert torch.is_tensor(tensor)

    x = torch.tensor([0.0])
    tensor = convert_tensor(x, device="cpu", non_blocking=True)
    assert torch.is_tensor(tensor)

    x = torch.tensor([0.0])
    tensor = convert_tensor(x, device="cpu", non_blocking=False)
    assert torch.is_tensor(tensor)

    x = [torch.tensor([0.0]), torch.tensor([0.0])]
    list_ = convert_tensor(x)
    assert isinstance(list_, list)
    assert torch.is_tensor(list_[0])
    assert torch.is_tensor(list_[1])

    x = (torch.tensor([0.0]), torch.tensor([0.0]))
    tuple_ = convert_tensor(x)
    assert isinstance(tuple_, tuple)
    assert torch.is_tensor(tuple_[0])
    assert torch.is_tensor(tuple_[1])

    Point = namedtuple("Point", ["x", "y"])
    x = Point(torch.tensor([0.0]), torch.tensor([0.0]))
    tuple_ = convert_tensor(x)
    assert isinstance(tuple_, Point)
    assert torch.is_tensor(tuple_[0])
    assert torch.is_tensor(tuple_[1])

    x = {"a": torch.tensor([0.0]), "b": torch.tensor([0.0])}
    dict_ = convert_tensor(x)
    assert isinstance(dict_, dict)
    assert torch.is_tensor(dict_["a"])
    assert torch.is_tensor(dict_["b"])

    assert convert_tensor("a") == "a"

    with pytest.raises(TypeError):
        convert_tensor(12345)


def test_to_onehot():
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    actual = to_onehot(indices, 4)
    expected = torch.eye(4, dtype=torch.uint8)
    assert actual.equal(expected)

    y = torch.randint(0, 21, size=(1000,))
    y_ohe = to_onehot(y, num_classes=21)
    y2 = torch.argmax(y_ohe, dim=1)
    assert y.equal(y2)

    y = torch.randint(0, 21, size=(4, 250, 255))
    y_ohe = to_onehot(y, num_classes=21)
    y2 = torch.argmax(y_ohe, dim=1)
    assert y.equal(y2)

    y = torch.randint(0, 21, size=(4, 150, 155, 4, 6))
    y_ohe = to_onehot(y, num_classes=21)
    y2 = torch.argmax(y_ohe, dim=1)
    assert y.equal(y2)


def test_dist_setup_logger():

    logger = setup_logger("trainer", level=logging.CRITICAL, distributed_rank=1)
    assert logger.level != logging.CRITICAL


def test_setup_logger(capsys, dirname):

    from ignite.engine import Engine, Events

    trainer = Engine(lambda e, b: None)
    evaluator = Engine(lambda e, b: None)

    fp = os.path.join(dirname, "log")
    assert len(trainer.logger.handlers) == 0
    trainer.logger.addHandler(logging.NullHandler())
    trainer.logger.addHandler(logging.NullHandler())
    trainer.logger.addHandler(logging.NullHandler())

    trainer.logger = setup_logger("trainer", filepath=fp)
    evaluator.logger = setup_logger("evaluator", filepath=fp)

    assert len(trainer.logger.handlers) == 2
    assert len(evaluator.logger.handlers) == 2

    @trainer.on(Events.EPOCH_COMPLETED)
    def _(_):
        evaluator.run([0, 1, 2])

    trainer.run([0, 1, 2, 3, 4, 5], max_epochs=5)

    captured = capsys.readouterr()
    err = captured.err.split("\n")

    with open(fp, "r") as h:
        data = h.readlines()

    for source in [err, data]:
        assert "trainer INFO: Engine run starting with max_epochs=5." in source[0]
        assert "evaluator INFO: Engine run starting with max_epochs=1." in source[2]


def _test_distrib_one_rank_only():

    def _test(barrier):
        # last rank
        rank = dist.get_world_size() - 1

        value = torch.tensor(0)

        @one_rank_only(rank=rank, barrier=barrier)
        def initialize():
            value.data = torch.tensor(100)

        initialize()

        value_list = [torch.tensor(0) for _ in range(dist.get_world_size())]

        dist.all_gather(tensor=value, tensor_list=value_list)

        for r in range(dist.get_world_size()):
            if r == rank:
                assert value_list[r].item() == 100
            else:
                assert value_list[r].item() == 0

    _test(barrier=True)
    _test(barrier=False)


def _test_distrib_one_rank_only_with_engine():

    def _test(barrier):
        engine = Engine(lambda e, b: b)

        batch_sum = torch.tensor(0)

        @engine.on(Events.ITERATION_COMPLETED)
        @one_rank_only(barrier=barrier)  # ie rank == 0
        def _(_):
            batch_sum.data += torch.tensor(engine.state.batch)

        engine.run([1, 2, 3], max_epochs=2)

        value_list = [torch.tensor(0) for _ in range(dist.get_world_size())]

        dist.all_gather(tensor=batch_sum, tensor_list=value_list)

        for r in range(dist.get_world_size()):
            if r == 0:
                assert value_list[r].item() == 12
            else:
                assert value_list[r].item() == 0

    _test(barrier=True)
    _test(barrier=False)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    _test_distrib_one_rank_only()
    _test_distrib_one_rank_only_with_engine()


@pytest.mark.multinode_distributed
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    _test_distrib_one_rank_only()
    _test_distrib_one_rank_only_with_engine()


@pytest.mark.multinode_distributed
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    _test_distrib_one_rank_only()
    _test_distrib_one_rank_only_with_engine()
