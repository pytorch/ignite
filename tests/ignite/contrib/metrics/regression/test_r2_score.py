import torch
import numpy as np
import pytest
from ignite.engine import Engine
from ignite.contrib.metrics.regression import R2Score
from sklearn.metrics import r2_score


def test_wrong_input_shapes():
    m = R2Score()

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


def test_r2_score():

    size = 51
    np_y_pred = np.random.rand(size,)
    np_y = np.random.rand(size,)

    m = R2Score()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    m.update((y_pred, y))

    assert r2_score(np_y, np_y_pred) == pytest.approx(m.compute())


def test_r2_score_2():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    m = R2Score()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    batch_size = 16
    n_iters = size // batch_size + 1
    for i in range(n_iters):
        idx = i * batch_size
        m.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

    assert r2_score(np_y, np_y_pred) == pytest.approx(m.compute())


def test_integration_r2_score_with_output_transform():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    batch_size = 15

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx:idx + batch_size]
        y_pred_batch = np_y_pred[idx:idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = R2Score(output_transform=lambda x: (x[1], x[2]))
    m.attach(engine, 'r2_score')

    data = list(range(size // batch_size))
    r_squared = engine.run(data, max_epochs=1).metrics['r2_score']

    assert r2_score(np_y, np_y_pred) == pytest.approx(r_squared)
