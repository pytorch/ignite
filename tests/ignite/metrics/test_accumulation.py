import torch
import numpy as np

import pytest

from ignite.metrics.accumulation import VariableAccumulation, Average, GeometricAverage
from ignite.exceptions import NotComputableError
from ignite.engine import Events, Engine


torch.manual_seed(15)


def test_variable_accumulation_wrong_inputs():

    with pytest.raises(TypeError, match=r"Argument op should be a callable"):
        VariableAccumulation(1)

    with pytest.raises(TypeError, match=r"Output should be a number or torch.Tensor,"):
        mean_acc = VariableAccumulation(lambda a, x: a + x)
        mean_acc.update((1, 2))

    with pytest.raises(TypeError, match=r"Output should be a number or torch.Tensor,"):
        mean_acc = VariableAccumulation(lambda a, x: a + x)
        mean_acc.update("a")


def test_variable_accumulation_mean_variable():

    mean_var = VariableAccumulation(lambda a, x: a + x)
    y_true = torch.rand(100)

    for y in y_true:
        mean_var.update(y)

    a, n = mean_var.compute()
    assert a.item() == pytest.approx(y_true.sum().item())
    assert n == len(y_true)

    mean_var = VariableAccumulation(lambda a, x: a + x)
    y_true = torch.rand(100, 10)
    for y in y_true:
        mean_var.update(y)

    a, n = mean_var.compute()
    assert a.numpy() == pytest.approx(y_true.sum(dim=0).numpy())
    assert n == len(y_true)


def test_average():

    with pytest.raises(NotComputableError):
        v = Average()
        v.compute()

    mean_var = Average()
    y_true = torch.rand(100) + torch.randint(0, 10, size=(100, )).float()

    for y in y_true:
        mean_var.update(y.item())

    m = mean_var.compute()
    assert m.item() == pytest.approx(y_true.mean().item())

    mean_var = Average()
    y_true = torch.rand(100, 10) + torch.randint(0, 10, size=(100, 10)).float()
    for y in y_true:
        mean_var.update(y)

    m = mean_var.compute()
    assert m.numpy() == pytest.approx(y_true.mean(dim=0).numpy())


def _geom_mean(t):
    np_t = t.numpy()
    return np.exp(np.mean(np.log(np_t), axis=0))


def test_geom_average():

    with pytest.raises(NotComputableError):
        v = GeometricAverage()
        v.compute()

    mean_var = GeometricAverage()
    y_true = torch.rand(100) + torch.randint(0, 10, size=(100,)).float()

    for y in y_true:
        mean_var.update(y.item())

    m = mean_var.compute()
    assert m.item() == pytest.approx(_geom_mean(y_true))

    mean_var = GeometricAverage()
    y_true = torch.rand(100, 10) + torch.randint(0, 10, size=(100, 10)).float()
    for y in y_true:
        mean_var.update(y)

    m = mean_var.compute()
    np.testing.assert_almost_equal(m.numpy(), _geom_mean(y_true), decimal=5)


def test_integration():

    def _test(metric_cls, true_result_fn):

        size = 100
        custom_variable = 10.0 + 5.0 * torch.rand(size, 12)

        def update_fn(engine, batch):
            return 0, custom_variable[engine.state.iteration - 1]

        engine = Engine(update_fn)

        custom_var_mean = metric_cls(output_transform=lambda output: output[1])
        custom_var_mean.attach(engine, 'agg_custom_var')

        state = engine.run([0] * size)
        np.testing.assert_almost_equal(state.metrics['agg_custom_var'].numpy(), true_result_fn(custom_variable),
                                       decimal=5)

        size = 100
        custom_variable = 10.0 + 5.0 * torch.rand(size)

        def update_fn(engine, batch):
            return 0, custom_variable[engine.state.iteration - 1].item()

        engine = Engine(update_fn)

        custom_var_mean = metric_cls(output_transform=lambda output: output[1])
        custom_var_mean.attach(engine, 'agg_custom_var')

        state = engine.run([0] * size)
        assert state.metrics['agg_custom_var'] == pytest.approx(true_result_fn(custom_variable))

    def _mean(y_true):
        return y_true.mean(dim=0).numpy()

    _test(Average, _mean)
    _test(GeometricAverage, _geom_mean)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib(local_rank, distributed_context_single_node):

    def test_distrib_variable_accumulation():

        import torch.distributed as dist

        device = "cuda:{}".format(local_rank)
        mean_var = VariableAccumulation(lambda a, x: a + x, device=device)
        y_true = torch.rand(100, device="cuda:{}".format(local_rank))

        for y in y_true:
            mean_var.update(y)

        dist.all_reduce(y_true)
        a, n = mean_var.compute()
        assert a.item() == pytest.approx(y_true.sum().item())
        assert n == len(y_true) * dist.get_world_size()
        # check if call compute twice
        a, n = mean_var.compute()
        assert a.item() == pytest.approx(y_true.sum().item())
        assert n == len(y_true) * dist.get_world_size()

        mean_var = VariableAccumulation(lambda a, x: a + x, device=device)
        y_true = torch.rand(50, 10, device="cuda:{}".format(local_rank))

        for y in y_true:
            mean_var.update(y)

        dist.all_reduce(y_true)
        a, n = mean_var.compute()
        np.testing.assert_almost_equal(a.cpu().numpy(),
                                       y_true.sum(dim=0).cpu().numpy(),
                                       decimal=5)
        assert n == len(y_true) * dist.get_world_size()
        a, n = mean_var.compute()
        np.testing.assert_almost_equal(a.cpu().numpy(),
                                       y_true.sum(dim=0).cpu().numpy(),
                                       decimal=5)
        assert n == len(y_true) * dist.get_world_size()

    test_distrib_variable_accumulation()

    def test_distrib_average():

        import torch.distributed as dist

        device = "cuda:{}".format(local_rank)

        with pytest.raises(NotComputableError):
            v = Average(device=device)
            v.compute()

        mean_var = Average(device=device)
        y_true = torch.rand(100) + torch.randint(0, 10, size=(100, )).float()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()

        dist.all_reduce(y_true)
        assert m.item() == pytest.approx(y_true.mean().item() / dist.get_world_size())

        mean_var = Average(device=device)
        y_true = torch.rand(100, 10) + torch.randint(0, 10, size=(100, 10)).float()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()

        dist.all_reduce(y_true)
        np.testing.assert_almost_equal(m.cpu().numpy(),
                                       y_true.mean(dim=0).cpu().numpy() / dist.get_world_size(),
                                       decimal=5)

    test_distrib_average()

    def test_distrib_geom_average():

        import torch.distributed as dist

        device = "cuda:{}".format(local_rank)

        with pytest.raises(NotComputableError):
            v = GeometricAverage(device=device)
            v.compute()

        mean_var = GeometricAverage(device=device)
        y_true = torch.rand(100) + torch.randint(0, 10, size=(100,)).float()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()
        log_y_true = torch.log(y_true)
        dist.all_reduce(log_y_true)
        assert m.item() == pytest.approx(torch.exp(log_y_true.mean(dim=0) / dist.get_world_size()).item())

        mean_var = GeometricAverage(device=device)
        y_true = torch.rand(100, 10) + torch.randint(0, 10, size=(100, 10)).float()
        y_true = y_true.to(device)

        for y in y_true:
            mean_var.update(y)

        m = mean_var.compute()
        log_y_true = torch.log(y_true)
        dist.all_reduce(log_y_true)
        np.testing.assert_almost_equal(m.cpu().numpy(),
                                       torch.exp(log_y_true.mean(dim=0) / dist.get_world_size()).cpu().numpy(),
                                       decimal=5)

    test_distrib_geom_average()

    def test_distrib_integration():

        import torch.distributed as dist

        device = "cuda:{}".format(local_rank)

        def _test(metric_cls, true_result_fn):

            size = 100
            custom_variable = 10.0 + 5.0 * torch.rand(size, 12)
            custom_variable = custom_variable.to(device)

            def update_fn(engine, batch):
                return 0, custom_variable[engine.state.iteration - 1]

            engine = Engine(update_fn)

            custom_var_mean = metric_cls(output_transform=lambda output: output[1],
                                         device=device)
            custom_var_mean.attach(engine, 'agg_custom_var')

            state = engine.run([0] * size)
            np.testing.assert_almost_equal(state.metrics['agg_custom_var'].cpu().numpy(),
                                           true_result_fn(custom_variable),
                                           decimal=5)

            size = 100
            custom_variable = 10.0 + 5.0 * torch.rand(size)
            custom_variable = custom_variable.to(device)

            def update_fn(engine, batch):
                return 0, custom_variable[engine.state.iteration - 1].item()

            engine = Engine(update_fn)

            custom_var_mean = metric_cls(output_transform=lambda output: output[1],
                                         device=device)
            custom_var_mean.attach(engine, 'agg_custom_var')

            state = engine.run([0] * size)
            assert state.metrics['agg_custom_var'] == pytest.approx(true_result_fn(custom_variable))

        def _mean(y_true):
            dist.all_reduce(y_true)
            return y_true.mean(dim=0).cpu().numpy() / dist.get_world_size()

        def _geom_mean(y_true):
            log_y_true = torch.log(y_true)
            dist.all_reduce(log_y_true)
            np_t = log_y_true.cpu().numpy()
            return np.exp(np.mean(np_t, axis=0) / dist.get_world_size())

        _test(Average, _mean)
        _test(GeometricAverage, _geom_mean)

    test_distrib_integration()
