import pytest

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import TopKCategoricalAccuracy


def test_zero_div():
    acc = TopKCategoricalAccuracy(2)
    with pytest.raises(NotComputableError):
        acc.compute()


def test_compute():
    acc = TopKCategoricalAccuracy(2)

    y_pred = torch.FloatTensor([[0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2]])
    y = torch.ones(2).long()
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 0.5

    acc.reset()
    y_pred = torch.FloatTensor([[0.4, 0.8, 0.2, 0.6], [0.8, 0.6, 0.4, 0.2]])
    y = torch.ones(2).long()
    acc.update((y_pred, y))
    assert isinstance(acc.compute(), float)
    assert acc.compute() == 1.0


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib(local_rank, distributed_context_single_node):

    import numpy as np

    def top_k_accuracy(y_true, y_pred, k=5, normalize=True):
        # Taken from
        # https://github.com/scikit-learn/scikit-learn/blob/4685cb5c50629aba4429f6701585f82fc3eee5f7/
        # sklearn/metrics/classification.py#L187
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        num_obs, num_labels = y_pred.shape
        idx = num_labels - k - 1
        counter = 0
        argsorted = np.argsort(y_pred, axis=1)
        for i in range(num_obs):
            if y_true[i] in argsorted[i, idx + 1:]:
                counter += 1
        if normalize:
            return counter / num_obs
        else:
            return counter

    def test_distrib_itegration():
        import torch.distributed as dist
        from ignite.engine import Engine

        torch.manual_seed(12)
        device = "cuda:{}".format(local_rank)

        def _test(n_epochs):
            n_iters = 100
            s = 16
            n_classes = 10

            offset = n_iters * s
            y_true = torch.randint(0, n_classes, size=(offset * dist.get_world_size(), )).to(device)
            y_preds = torch.rand(offset * dist.get_world_size(), n_classes).to(device)

            def update(engine, i):
                return y_preds[i * s + local_rank * offset:(i + 1) * s + local_rank * offset, :], \
                    y_true[i * s + local_rank * offset:(i + 1) * s + local_rank * offset]

            engine = Engine(update)

            k = 5
            acc = TopKCategoricalAccuracy(k=k, device=device)
            acc.attach(engine, "acc")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=n_epochs)

            assert "acc" in engine.state.metrics
            res = engine.state.metrics['acc']
            if isinstance(res, torch.Tensor):
                res = res.cpu().numpy()

            true_res = top_k_accuracy(y_true.cpu().numpy(),
                                      y_preds.cpu().numpy(), k=k)

            assert pytest.approx(res) == true_res

        for _ in range(5):
            _test(n_epochs=1)
            _test(n_epochs=2)

    test_distrib_itegration()
