import torch
import numpy as np
import pytest
from ignite.engine import Engine
from ignite.contrib.metrics.regression import MedianRelativeAbsoluteError


def test_wrong_input_shapes():
    m = MedianRelativeAbsoluteError()

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


def test_median_relative_absolute_error():

    # See https://github.com/torch/torch7/pull/182
    # For even number of elements, PyTorch returns middle element
    # NumPy returns average of middle elements
    # Size of dataset will be odd for these tests

    size = 51
    np_y_pred = np.random.rand(size,)
    np_y = np.random.rand(size,)
    np_median_absolute_relative_error = np.median(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean()))

    m = MedianRelativeAbsoluteError()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    m.update((y_pred, y))

    assert np_median_absolute_relative_error == pytest.approx(m.compute())


def test_median_relative_absolute_error_2():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)
    np_median_absolute_relative_error = np.median(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean()))

    m = MedianRelativeAbsoluteError()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    batch_size = 16
    n_iters = size // batch_size + 1
    for i in range(n_iters + 1):
        idx = i * batch_size
        m.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

    assert np_median_absolute_relative_error == pytest.approx(m.compute())


def test_integration_median_relative_absolute_error_with_output_transform():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)
    np_median_absolute_relative_error = np.median(np.abs(np_y - np_y_pred) / np.abs(np_y - np_y.mean()))

    batch_size = 15

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx:idx + batch_size]
        y_pred_batch = np_y_pred[idx:idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = MedianRelativeAbsoluteError(output_transform=lambda x: (x[1], x[2]))
    m.attach(engine, 'median_absolute_relative_error')

    data = list(range(size // batch_size))
    median_absolute_relative_error = engine.run(data, max_epochs=1).metrics['median_absolute_relative_error']

    assert np_median_absolute_relative_error == pytest.approx(median_absolute_relative_error)
