# -*- coding: utf-8 -*-
import json
from mock import call, patch
import numpy as np
import os
import tempfile
import time

from ignite.contrib.handlers import VisdomLogger
from ignite.engine import Engine


def process_function(engine, batch):
    a = 1
    engine.state.metrics['a'] = a
    return a


@patch('visdom.Visdom')
def test_events(vis):
    """Test calls to visdom logger."""

    #
    # Create replies to visdom calls.
    #
    vis.check_connection.return_value = True
    vis.line.return_value = 1234

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(process_function)

    visdom_logger = VisdomLogger(
        vis=vis
    )

    loss_win = visdom_logger.create_window(
        window_title="Training Loss",
        xlabel="Iteration",
        ylabel="Loss"
    )
    loss_win.attach(
        engine=engine,
        output_transform=lambda x: {"loss": x},
    )

    engine.run(loader, max_epochs=n_epochs)

    #
    # Sleep to allow the visdom logger thread to finish.
    #
    time.sleep(0.1)

    line_calls = [
        call(
            X=np.array([0]),
            Y=np.array([1]),
            env='main',
            name='loss',
            opts={
                'title': 'Training Loss',
                'xlabel': 'Iteration',
                'ylabel': 'Loss',
                'showlegend': False
            },
            update=None,
            win=None
        ),
        call(
            X=np.array([1]),
            Y=np.array([1]),
            env='main',
            name='loss',
            opts={
                'title': 'Training Loss',
                'xlabel': 'Iteration',
                'ylabel': 'Loss',
                'showlegend': False
            },
            update='append',
            win=1234
        )
    ]

    assert vis.line.call_args_list == line_calls


def test_log_file():
    """Test creation of the visdom log file."""

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(process_function)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        log_filename = f.name

    visdom_logger = VisdomLogger(log_to_filename=log_filename)

    loss_win = visdom_logger.create_window(
        window_title="Training Loss",
        xlabel="Iteration",
        ylabel="Loss"
    )
    loss_win.attach(
        engine=engine,
        output_transform=lambda x: {"loss": x},
    )

    engine.run(loader, max_epochs=n_epochs)

    #
    # Sleep to allow the visdom logger thread to finish.
    #
    time.sleep(0.1)

    with open(log_filename) as f:
        visdom_log = [json.loads(l.strip()) for l in f.readlines()]

    os.remove(log_filename)

    #
    # Assert the events in the log file.
    # The actual contents of the events are not checked as these depend on the
    # installed visdom version.
    #
    assert len(visdom_log) == 2
    assert visdom_log[0][0] == 'events'
    assert visdom_log[1][0] == 'update'
