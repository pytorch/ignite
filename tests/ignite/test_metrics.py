import torch
from pytest import approx

from ignite.history import History
from ignite.metrics import (
    binary_accuracy,
    categorical_accuracy,
    top_k_categorical_accuracy,
    mean_squared_error,
    mean_absolute_error
)


def test_binary_accuracy():
    history = History()
    history.append((torch.FloatTensor([[0.8], [0.6]]), torch.LongTensor([1, 0])))
    history.append((torch.FloatTensor([[0.4], [0.2]]), torch.LongTensor([1, 0])))
    result = binary_accuracy(history)
    assert result == approx(0.5)

    history = History()
    history.append({'y_pred': torch.FloatTensor([[0.8], [0.6]]), 'y': torch.LongTensor([1, 0])})
    history.append({'y_pred': torch.FloatTensor([[0.4], [0.2]]), 'y': torch.LongTensor([1, 0])})
    result = binary_accuracy(history, transform=lambda x: (x['y_pred'], x['y']))

    assert result == approx(0.5)


def test_categorical_accuracy():
    history = History()
    history.append((torch.FloatTensor([[2, 1], [4, 3]]), torch.LongTensor([1, 0])))
    history.append((torch.FloatTensor([[6, 5], [8, 7]]), torch.LongTensor([1, 0])))
    result = categorical_accuracy(history)
    assert result == approx(0.5)

    history = History()
    history.append({'y_pred': torch.FloatTensor([[2, 1], [4, 3]]), 'y': torch.LongTensor([1, 0])})
    history.append({'y_pred': torch.FloatTensor([[6, 5], [8, 7]]), 'y': torch.LongTensor([1, 0])})
    result = categorical_accuracy(history, transform=lambda x: (x['y_pred'], x['y']))
    assert result == approx(0.5)


def test_top_k_categorical_accuracy():
    history = History()
    history.append((torch.FloatTensor([[3, 2, 1], [6, 5, 4]]), torch.LongTensor([1, 0])))
    history.append((torch.FloatTensor([[9, 8, 7], [12, 11, 10]]), torch.LongTensor([1, 0])))
    top_2_result = top_k_categorical_accuracy(history, k=2)
    top_1_result = top_k_categorical_accuracy(history, k=1)
    assert top_2_result == approx(1.0)
    assert top_1_result == approx(0.5)
    assert top_1_result == approx(categorical_accuracy(history))

    history = History()
    history.append({
        'y_pred': torch.FloatTensor([[3, 2, 1], [6, 5, 4]]),
        'y': torch.LongTensor([1, 0])
    })
    history.append({
        'y_pred': torch.FloatTensor([[9, 8, 7], [12, 11, 10]]),
        'y': torch.LongTensor([1, 0])
    })
    top_2_result = top_k_categorical_accuracy(history,
                                              k=2,
                                              transform=lambda x: (x['y_pred'], x['y']))
    top_1_result = top_k_categorical_accuracy(history,
                                              k=1,
                                              transform=lambda x: (x['y_pred'], x['y']))
    assert top_2_result == approx(1.0)
    assert top_1_result == approx(0.5)
    assert top_1_result == approx(categorical_accuracy(history,
                                                       transform=lambda x: (x['y_pred'], x['y'])))


def test_mean_squared_error():
    history = History()
    history.append((torch.FloatTensor([[4.5], [4.0]]), torch.FloatTensor([5.0, 3.5])))
    history.append((torch.FloatTensor([[3.5], [3.0]]), torch.FloatTensor([3.0, 3.5])))
    result = mean_squared_error(history)
    assert result == approx(0.25)

    history = History()
    history.append({
        'y_pred': torch.FloatTensor([[4.5], [4.0]]),
        'y': torch.FloatTensor([5.0, 3.5])
    })
    history.append({
        'y_pred': torch.FloatTensor([[3.5], [3.0]]),
        'y': torch.FloatTensor([3.0, 3.5])
    })
    result = mean_squared_error(history, transform=lambda x: (x['y_pred'], x['y']))
    assert result == approx(0.25)


def test_mean_absolute_error():
    history = History()
    history.append((torch.FloatTensor([[4.5], [4.0]]), torch.FloatTensor([5.0, 3.5])))
    history.append((torch.FloatTensor([[3.5], [3.0]]), torch.FloatTensor([3.0, 3.5])))
    result = mean_absolute_error(history)
    assert result == approx(0.5)

    history = History()
    history.append({
        'y_pred': torch.FloatTensor([[4.5], [4.0]]),
        'y': torch.FloatTensor([5.0, 3.5])
    })
    history.append({
        'y_pred': torch.FloatTensor([[3.5], [3.0]]),
        'y': torch.FloatTensor([3.0, 3.5])
    })
    result = mean_absolute_error(history, transform=lambda x: (x['y_pred'], x['y']))
    assert result == approx(0.5)
