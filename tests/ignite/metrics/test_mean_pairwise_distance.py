import pytest
from pytest import approx

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import MeanPairwiseDistance


def test_zero_div():
    mpd = MeanPairwiseDistance()
    with pytest.raises(NotComputableError):
        mpd.compute()


def test_compute():
    mpd = MeanPairwiseDistance()

    y_pred = torch.Tensor([[3.0, 4.0], [-3.0, -4.0]])
    y = torch.zeros(2, 2)
    mpd.update((y_pred, y))
    assert isinstance(mpd.compute(), float)
    assert mpd.compute() == approx(5.0)

    mpd.reset()
    y_pred = torch.Tensor([[4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, -4.0, -4.0]])
    y = torch.zeros(2, 4)
    mpd.update((y_pred, y))
    assert isinstance(mpd.compute(), float)
    assert mpd.compute() == approx(8.0)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib(local_rank, distributed_context_single_node):

    def test_distrib_itegration():
        import numpy as np
        import torch.distributed as dist
        from ignite.engine import Engine

        torch.manual_seed(12)

        device = "cuda:{}".format(local_rank)

        n_iters = 100
        s = 50
        offset = n_iters * s

        y_true = torch.rand(offset * dist.get_world_size(), 10).to(device)
        y_preds = torch.rand(offset * dist.get_world_size(), 10).to(device)

        def update(engine, i):
            return y_preds[i * s + offset * local_rank:(i + 1) * s + offset * local_rank, ...], \
                y_true[i * s + offset * local_rank:(i + 1) * s + offset * local_rank, ...]

        engine = Engine(update)

        m = MeanPairwiseDistance(device=device)
        m.attach(engine, "mpwd")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "mpwd" in engine.state.metrics
        res = engine.state.metrics['mpwd']

        true_res = []
        for i in range(n_iters * dist.get_world_size()):
            true_res.append(
                torch.pairwise_distance(y_true[i * s:(i + 1) * s, ...],
                                        y_preds[i * s:(i + 1) * s, ...],
                                        p=m._p, eps=m._eps).cpu().numpy()
            )
        true_res = np.array(true_res).ravel()
        print(local_rank, true_res.shape)
        true_res = true_res.mean()

        assert pytest.approx(res) == true_res

    test_distrib_itegration()
