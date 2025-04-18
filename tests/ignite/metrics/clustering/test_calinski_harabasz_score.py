from typing import Tuple

import numpy as np
import pytest

import torch
from sklearn.metrics import calinski_harabasz_score
from torch import Tensor

from ignite import distributed as idist
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.clustering import CalinskiHarabaszScore


def test_zero_sample():
    with pytest.raises(
        NotComputableError, match="CalinskiHarabaszScore must have at least one example before it can be computed"
    ):
        metric = CalinskiHarabaszScore()
        metric.compute()


def test_wrong_output_shape():
    wrong_features = torch.zeros(4, dtype=torch.float)
    correct_features = torch.zeros(4, 3, dtype=torch.float)
    wrong_labels = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    correct_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    with pytest.raises(ValueError, match=r"Features should be of shape \(batch_size, n_targets\)"):
        metric = CalinskiHarabaszScore()
        metric.update((wrong_features, correct_labels))

    with pytest.raises(ValueError, match=r"Labels should be of shape \(batch_size, \)"):
        metric = CalinskiHarabaszScore()
        metric.update((correct_features, wrong_labels))


def test_wrong_output_dtype():
    wrong_features = torch.zeros(4, 3, dtype=torch.long)
    correct_features = torch.zeros(4, 3, dtype=torch.float)
    wrong_labels = torch.tensor([0, 0, 1, 1], dtype=torch.float)
    correct_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    with pytest.raises(ValueError, match=r"Incoherent types between input features and stored features"):
        metric = CalinskiHarabaszScore()
        metric.update((correct_features, correct_labels))
        metric.update((wrong_features, correct_labels))

    with pytest.raises(ValueError, match=r"Incoherent types between input labels and stored labels"):
        metric = CalinskiHarabaszScore()
        metric.update((correct_features, correct_labels))
        metric.update((correct_features, wrong_labels))


@pytest.fixture(params=list(range(2)))
def test_case(request):
    N = 100
    NDIM = 10
    BS = 10

    # well-clustered case
    random_order = torch.from_numpy(np.random.permutation(N * 3))
    x1 = torch.cat(
        [
            torch.normal(-5.0, 1.0, size=(N, NDIM)),
            torch.normal(5.0, 1.0, size=(N, NDIM)),
            torch.normal(0.0, 1.0, size=(N, NDIM)),
        ]
    ).float()[random_order]
    y1 = torch.tensor([0] * N + [1] * N + [2] * N, dtype=torch.long)[random_order]

    # poorly-clustered case
    x2 = torch.cat(
        [
            torch.normal(-1.0, 1.0, size=(N, NDIM)),
            torch.normal(0.0, 1.0, size=(N, NDIM)),
            torch.normal(1.0, 1.0, size=(N, NDIM)),
        ]
    ).float()
    y2 = torch.from_numpy(np.random.choice(3, size=N * 3)).long()

    return [
        (x1, y1, BS),
        (x2, y2, BS),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_integration(n_times: int, test_case: Tuple[Tensor, Tensor, Tensor], available_device):
    features, labels, batch_size = test_case

    np_features = features.numpy()
    np_labels = labels.numpy()

    def update_fn(engine: Engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        feature_batch = np_features[idx : idx + batch_size]
        label_batch = np_labels[idx : idx + batch_size]
        return torch.from_numpy(feature_batch), torch.from_numpy(label_batch)

    engine = Engine(update_fn)

    m = CalinskiHarabaszScore(device=available_device)
    assert m._device == torch.device(available_device)

    m.attach(engine, "calinski_harabasz_score")

    data = list(range(np_features.shape[0] // batch_size))
    s = engine.run(data, max_epochs=1).metrics["calinski_harabasz_score"]

    np_ans = calinski_harabasz_score(np_features, np_labels)

    assert pytest.approx(np_ans, rel=1e-5) == s


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_compute(self):
        rank = idist.get_rank()
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        torch.manual_seed(10 + rank)
        for metric_device in metric_devices:
            m = CalinskiHarabaszScore(device=metric_device)

            random_order = torch.from_numpy(np.random.permutation(200))
            features = torch.cat([torch.normal(-1.0, 1.0, size=(100, 10)), torch.normal(1.0, 1.0, size=(100, 10))]).to(
                device
            )[random_order]
            labels = torch.tensor([0] * 100 + [1] * 100, dtype=torch.long, device=device)[random_order]

            m.update((features, labels))

            features = idist.all_gather(features)
            labels = idist.all_gather(labels)

            np_features = features.cpu().numpy()
            np_labels = labels.cpu().numpy()

            np_ans = calinski_harabasz_score(np_features, np_labels)

            assert pytest.approx(np_ans, rel=1e-5) == m.compute()

    @pytest.mark.parametrize("n_epochs", [1, 2])
    def test_integration(self, n_epochs: int):
        tol = 1e-5
        rank = idist.get_rank()
        device = idist.device()
        metric_devices = [torch.device("cpu")]
        if device.type != "xla":
            metric_devices.append(device)

        n_iters = 80
        batch_size = 16

        for metric_device in metric_devices:
            torch.manual_seed(12 + rank)

            cluster_size = n_iters * batch_size // 2
            random_order = torch.from_numpy(np.random.permutation(n_iters * batch_size))
            features = torch.cat(
                [torch.normal(-1.0, 1.0, size=(cluster_size, 10)), torch.normal(1.0, 1.0, size=(cluster_size, 10))]
            ).to(device)[random_order]
            labels = torch.tensor([0] * cluster_size + [1] * cluster_size, dtype=torch.long, device=device)[
                random_order
            ]

            engine = Engine(
                lambda e, i: (
                    features[i * batch_size : (i + 1) * batch_size],
                    labels[i * batch_size : (i + 1) * batch_size],
                )
            )

            chs = CalinskiHarabaszScore(device=metric_device)
            chs.attach(engine, "calinski_harabasz_score")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=n_epochs)

            features = idist.all_gather(features)
            labels = idist.all_gather(labels)

            assert "calinski_harabasz_score" in engine.state.metrics

            res = engine.state.metrics["calinski_harabasz_score"]

            np_labels = labels.cpu().numpy()
            np_features = features.cpu().numpy()

            np_ans = calinski_harabasz_score(np_features, np_labels)

            assert pytest.approx(np_ans, rel=tol) == res
