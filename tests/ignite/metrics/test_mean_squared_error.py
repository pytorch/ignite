from ignite.exceptions import NotComputableError
from ignite.metrics import MeanSquaredError
import pytest
import torch


def test_zero_div():
    mse = MeanSquaredError()
    with pytest.raises(NotComputableError):
        mse.compute()


def test_compute():
    mse = MeanSquaredError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    mse.update((y_pred, y))
    assert isinstance(mse.compute(), float)
    assert mse.compute() == 4.0

    mse.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    mse.update((y_pred, y))
    assert isinstance(mse.compute(), float)
    assert mse.compute() == 9.0


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
        y_preds = torch.ones(offset * dist.get_world_size(), dtype=torch.float).to(device)

        def update(engine, i):
            return y_preds[i * s + offset * local_rank:(i + 1) * s + offset * local_rank], \
                y_true[i * s + offset * local_rank:(i + 1) * s + offset * local_rank]

        engine = Engine(update)

        m = MeanSquaredError(device=device)
        m.attach(engine, "mse")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "mse" in engine.state.metrics
        res = engine.state.metrics['mse']

        true_res = np.mean(np.power((y_true - y_preds).cpu().numpy(), 2.0))

        assert pytest.approx(res) == true_res

    test_distrib_itegration()
