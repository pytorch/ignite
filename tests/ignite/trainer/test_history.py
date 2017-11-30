import numpy as np

from ignite.trainer.history import History


def test_history_clear():
    history = History()
    for i in range(5):
        history.append(i)
    history.clear()
    assert len(history) == 0


def test_history_simple_moving_average():
    history = History()
    history.append(1)
    history.append(3)
    history.append(4)
    sma = history.simple_moving_average(window_size=3)
    np.testing.assert_almost_equal(sma, 8 / 3.0)

    history = History()
    history.append({'loss': 1, 'other': 2})
    history.append({'loss': 3, 'other': 6})
    history.append({'loss': 4, 'other': 5})
    sma = history.simple_moving_average(window_size=3, transform=lambda x: x['loss'])
    np.testing.assert_almost_equal(sma, 8 / 3.0)


def test_weighted_moving_average():
    history = History()
    history.append(1)
    history.append(3)
    history.append(4)
    wma = history.weighted_moving_average(window_size=3, weights=[0.6, 0.8, 1.0])
    np.testing.assert_almost_equal(wma, (0.6 + (0.8 * 3) + 4) / 2.4)

    history = History()
    history.append({'loss': 1, 'other': 2})
    history.append({'loss': 3, 'other': 6})
    history.append({'loss': 4, 'other': 5})
    wma = history.weighted_moving_average(window_size=3, weights=[0.6, 0.8, 1.0], transform=lambda x: x['loss'])
    np.testing.assert_almost_equal(wma, (0.6 + (0.8 * 3) + 4) / 2.4)


def test_exponential_moving_average():
    history = History()
    history.append(1)
    history.append(3)
    history.append(4)
    alpha = 0.3
    epa = history.exponential_moving_average(window_size=3, alpha=alpha)
    expected_epa = (4 + (1 - alpha) * 3 + (1 - alpha)**2 * 1) / (1 + (1 - alpha) + (1 - alpha)**2)
    np.testing.assert_almost_equal(epa, expected_epa)

    history = History()
    history.append({'loss': 1, 'other': 2})
    history.append({'loss': 3, 'other': 6})
    history.append({'loss': 4, 'other': 5})
    epa = history.exponential_moving_average(window_size=3, alpha=alpha, transform=lambda x: x['loss'])
    expected_epa = (4 + (1 - alpha) * 3 + (1 - alpha)**2 * 1) / (1 + (1 - alpha) + (1 - alpha)**2)
    np.testing.assert_almost_equal(epa, expected_epa)
