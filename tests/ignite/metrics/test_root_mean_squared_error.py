from ignite.exceptions import NotComputableError
from ignite.metrics import RootMeanSquaredError
import pytest
import torch


def test_zero_div():
    rmse = RootMeanSquaredError()
    with pytest.raises(NotComputableError):
        rmse.compute()


def test_compute():
    rmse = RootMeanSquaredError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 2.0

    rmse.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 3.0


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib(local_rank, distributed_context_single_node):

    def test_distrib_itegration():
        import numpy as np
        import torch.distributed as dist
        from ignite.engine import Engine

        device = "cuda:{}".format(local_rank)

        n_iters = 100
        s = 50
        offset = n_iters * s

        y_true = torch.arange(0, offset * dist.get_world_size(), dtype=torch.float).to(device)
        y_preds = (local_rank + 1) * torch.ones(offset, dtype=torch.float).to(device)

        def update(engine, i):
            return y_preds[i * s:(i + 1) * s], y_true[i * s + offset * local_rank:(i + 1) * s + offset * local_rank]

        engine = Engine(update)

        m = RootMeanSquaredError(device=device)
        m.attach(engine, "rmse")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "rmse" in engine.state.metrics
        res = engine.state.metrics['rmse']

        y_preds_full = []
        for i in range(dist.get_world_size()):
            y_preds_full.append((i + 1) * torch.ones(offset))
        y_preds_full = torch.stack(y_preds_full).to(device).flatten()

        true_res = np.sqrt(np.mean(np.square((y_true - y_preds_full).cpu().numpy())))

        assert pytest.approx(res) == true_res

    test_distrib_itegration()
