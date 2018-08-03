# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytest
from mock import MagicMock

from ignite.contrib.engines import create_supervised_tbptt_trainer, Tbptt_Events
from ignite.contrib.engines import _detach_hidden


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


def _test_create_supervised_tbptt_trainer(device):
    # Defining dummy recurrent model with zero weights
    model = nn.RNN(1, 1, bias=False)
    for p in model.parameters():
        p.data.zero_()

    # Defning optimizer and trainer
    optimizer = optim.SGD(model.parameters(), 1)
    trainer = create_supervised_tbptt_trainer(
        model,
        optimizer,
        F.mse_loss,
        tbtt_step=2,
        device=device
    )

    # Adding two mock handles to the trainer to monitor that TBPTT events are
    # called correctly
    handle_started = MagicMock()
    trainer.add_event_handler(
        Tbptt_Events.TIME_ITERATION_STARTED,
        handle_started
    )
    handle_completed = MagicMock()
    trainer.add_event_handler(
        Tbptt_Events.TIME_ITERATION_COMPLETED,
        handle_completed
    )

    # Fake data
    X = torch.ones(6, 2, 1)
    y = X = torch.ones(6, 2, 1)
    data = [(X, y)]

    # Running trainer
    trainer.run(data)

    # Verifications
    assert handle_started.call_count == 3
    assert handle_completed.call_count == 3

    # If tbptt is not use (one gradient update), the hidden to hidden weight
    # should stay zero
    assert not model.weight_hh_l0.item() == pytest.approx(0)


def test_create_supervised_tbptt_trainer_with_cpu():
    _test_create_supervised_tbptt_trainer("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_create_supervised_tbptt_trainer_with_gpu():
    _test_create_supervised_tbptt_trainer("gpu")
