import torch


def _convert_history(history, transform):
    y_preds, ys = zip(*map(transform, history))
    y_pred = torch.cat(y_preds, dim=0)
    y = torch.cat(ys, dim=0)
    return y_pred, y


def binary_accuracy(history, transform=lambda x: x):
    """
    Calculates the binary accuracy

    Args:
        history (History): the history used to compute the metric
        transform (callable, optional): takes a history item and converts it into the form
            (y_pred, y) (default: identity)

    Returns:
        float: the accuracy over the entire history
    """
    y_pred, y = _convert_history(history, transform)
    correct = torch.eq(torch.round(y_pred).type(torch.LongTensor), y)
    return torch.mean(correct.type(torch.FloatTensor))


def categorical_accuracy(history, transform=lambda x: x):
    """
    Calculates the categorical accuracy

    Args:
        history (History): the history used to compute the metric
        transform (callable, optional): takes a history item and converts it into the form
            (y_pred, y) (default: identity)

    Returns:
        float: the accuracy over the entire history
    """
    y_pred, y = _convert_history(history, transform)
    indices = torch.max(y_pred, 1)[1]
    correct = torch.eq(indices, y)
    return torch.mean(correct.type(torch.FloatTensor))


def top_k_categorical_accuracy(history, k=5, transform=lambda x: x):
    """
    Calculates the top-k categorical accuracy

    Args:
        history (History): the history used to compute the metric
        transform (callable, optional): takes a history item and converts it into the form
            (y_pred, y) (default: identity)

    Returns:
        float: the accuracy over the entire history
    """
    y_pred, y = _convert_history(history, transform)
    sorted_indices = torch.topk(y_pred, k, dim=1)[1]
    expanded_y = y.view(-1, 1).expand(-1, k)
    correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
    return torch.mean(correct.type(torch.FloatTensor))


def mean_squared_error(history, transform=lambda x: x):
    """
    Calculates the mean squared error

    Args:
        history (History): the history used to compute the metric
        transform (callable, optional): takes a history item and converts it into the form
            (y_pred, y) (default: identity)

    Returns:
        float: the error over the entire history
    """
    y_pred, y = _convert_history(history, transform)
    squared_error = torch.pow(y_pred - y.view_as(y_pred), 2)
    return torch.mean(squared_error)


def mean_absolute_error(history, transform=lambda x: x):
    """
    Calculates the mean absolute error

    Args:
        history (History): the history used to compute the metric
        transform (callable, optional): takes a history item and converts it into the form
            (y_pred, y) (default: identity)

    Returns:
        float: the error over the entire history
    """
    y_pred, y = _convert_history(history, transform)
    absolute_error = torch.abs(y_pred - y.view_as(y_pred))
    return torch.mean(absolute_error)
