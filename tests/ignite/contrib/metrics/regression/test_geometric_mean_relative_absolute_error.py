import torch
import numpy as np
import pytest
from ignite.engine import Engine
from ignite.contrib.metrics.regression import GeometricMeanRelativeAbsoluteError


def test_wrong_input_shapes():
    m = GeometricMeanRelativeAbsoluteError()

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2),
                  torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1),
                  torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2),
                  torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,),
                  torch.rand(4, 1, 2)))


def test_geometric_mean_relative_absolute_error():
    size = 51
    np_y_pred = np.random.rand(size,)
    np_y = np.random.rand(size,)
    np_gmrae = np.exp(np.log(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean())).mean())

    m = GeometricMeanRelativeAbsoluteError()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    m.update((y_pred, y))

    assert np_gmrae == pytest.approx(m.compute())


def test_geometric_mean_relative_absolute_error_2():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    np_y_sum = 0
    num_examples = 0
    num_sum_of_errors = 0
    np_gmrae = 0

    m = GeometricMeanRelativeAbsoluteError()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    n_iters = 15
    batch_size = size // n_iters
    for i in range(n_iters + 1):
        idx = i * batch_size
        np_y_i = np_y[idx: idx + batch_size]
        np_y_pred_i = np_y_pred[idx: idx + batch_size]

        np_y_sum += np_y_i.sum()
        num_examples += np_y_i.shape[0]
        np_mean = np_y_sum / num_examples

        np_gmrae += np.log(np.abs(np_y_i - np_y_pred_i) / np.abs(np_y_i - np_mean)).sum()
        m.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

    assert np.exp(np_gmrae / num_examples) == pytest.approx(m.compute())


def test_integration_geometric_mean_relative_absolute_error_with_output_transform():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    np_y_sum = 0
    num_examples = 0
    num_sum_of_errors = 0
    np_gmrae = 0

    n_iters = 15
    batch_size = size // n_iters
    for i in range(n_iters + 1):
        idx = i * batch_size
        np_y_i = np_y[idx: idx + batch_size]
        np_y_pred_i = np_y_pred[idx: idx + batch_size]

        np_y_sum += np_y_i.sum()
        num_examples += np_y_i.shape[0]
        np_mean = np_y_sum / num_examples

        np_gmrae += np.log(np.abs(np_y_i - np_y_pred_i) / np.abs(np_y_i - np_mean)).sum()

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx:idx + batch_size]
        y_pred_batch = np_y_pred[idx:idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = GeometricMeanRelativeAbsoluteError(output_transform=lambda x: (x[1], x[2]))
    m.attach(engine, 'geometric_mean_relative_absolute_error')

    data = list(range(size // batch_size))
    gmrae = engine.run(data, max_epochs=1).metrics['geometric_mean_relative_absolute_error']

    assert np.exp(np_gmrae / num_examples) == pytest.approx(m.compute())
