# coding: utf-8

import unittest.mock as mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.contrib.engines import create_supervised_tbptt_trainer, Tbptt_Events
from ignite.contrib.engines.tbptt import _detach_hidden


def test_detach_hidden_RNN():
    # Create hidden vector (in tuple)
    X = torch.ones(2, 3, 4)
    model = nn.RNN(4, 1)
    _, hidden = model(X)

    # Function to test
    hidden_ = _detach_hidden(hidden)

    assert hidden_.grad_fn is None  # properly detached
    assert (hidden == hidden_).all().item() == 1  # Equal values


def test_detach_hidden_LSTM():
    # Create hidden vector (in tuple)
    X = torch.ones(2, 3, 4)
    model = nn.LSTM(4, 1)
    _, hidden = model(X)

    # Function to test
    hidden_ = _detach_hidden(hidden)

    for h, h_ in zip(hidden, hidden_):
        assert h_.grad_fn is None  # properly detached
        assert (h == h_).all().item() == 1  # Equal values


def test_detach_hidden_raise():
    with pytest.raises(TypeError):
        _detach_hidden(0)


@mock.patch("ignite.contrib.engines.tbptt._detach_hidden")
def test_create_supervised_tbptt_trainer_callcounts(mock_detach_hidden):
    # Mocking objects
    model = mock.MagicMock()
    # Necessary to unpack output
    model.return_value = (1, 1)
    optimizer = mock.MagicMock()
    loss = mock.MagicMock()

    trainer = create_supervised_tbptt_trainer(model, optimizer, loss, tbtt_step=2)

    # Adding two mock handles to the trainer to monitor that TBPTT events are
    # called correctly
    handle_started = mock.MagicMock()
    trainer.add_event_handler(Tbptt_Events.TIME_ITERATION_STARTED, handle_started)
    handle_completed = mock.MagicMock()
    trainer.add_event_handler(Tbptt_Events.TIME_ITERATION_COMPLETED, handle_completed)

    # Fake data
    X = torch.ones(6, 2, 1)
    y = torch.ones(6, 2, 1)
    data = [(X, y)]

    # Running trainer
    trainer.run(data)

    # Verifications
    assert handle_started.call_count == 3
    assert handle_completed.call_count == 3
    assert mock_detach_hidden.call_count == 2
    assert model.call_count == 3
    assert loss.call_count == 3
    assert optimizer.zero_grad.call_count == 3
    assert optimizer.step.call_count == 3
    n_args_tuple = tuple(len(args) for args, kwargs in model.call_args_list)
    assert n_args_tuple == (1, 2, 2)


def _test_create_supervised_tbptt_trainer(device):
    # Defining dummy recurrent model with zero weights
    model = nn.RNN(1, 1, bias=False)
    model.to(device)  # Move model before creating optimizer
    for p in model.parameters():
        p.data.zero_()

    # Set some mock on forward to monitor
    forward_mock = mock.MagicMock()
    forward_mock.return_value = None
    model.register_forward_hook(forward_mock)

    # Defning optimizer and trainer
    optimizer = optim.SGD(model.parameters(), 1)
    trainer = create_supervised_tbptt_trainer(model, optimizer, F.mse_loss, tbtt_step=2, device=device)

    # Fake data
    X = torch.ones(6, 2, 1)
    y = torch.ones(6, 2, 1)
    data = [(X, y)]

    # Running trainer
    trainer.run(data)

    # If tbptt is not use (one gradient update), the hidden to hidden weight
    # should stay zero
    assert not model.weight_hh_l0.item() == pytest.approx(0)

    # Cheking forward calls
    assert forward_mock.call_count == 3
    for i in range(3):
        inputs = forward_mock.call_args_list[i][0][1]
        if i == 0:
            assert len(inputs) == 1
        else:
            assert len(inputs) == 2
            x, h = inputs
            assert h.is_leaf


def test_create_supervised_tbptt_trainer_with_cpu():
    _test_create_supervised_tbptt_trainer("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_tbptt_trainer_on_cuda():
    _test_create_supervised_tbptt_trainer("cuda")
