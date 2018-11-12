# -*- coding: utf-8 -*-
import json
import os
import pytest

from ignite.contrib.handlers import VisdomLogger
from ignite.engine import Engine
import tempfile


expected_log = [['events',
  {'data': [{'x': [0],
     'y': [1],
     'name': 'loss',
     'type': 'scatter',
     'mode': 'lines',
     'textposition': 'right',
     'marker': {'size': 10,
      'symbol': 'dot',
      'line': {'color': '#000000', 'width': 0.5}}}],
   'eid': 'main',
   'layout': {'showlegend': False,
    'title': 'Training Loss',
    'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},
    'xaxis': {'title': 'Iteration'},
    'yaxis': {'title': 'Loss'}},
   'opts': {'title': 'Training Loss',
    'xlabel': 'Iteration',
    'ylabel': 'Loss',
    'showlegend': False,
    'markers': False,
    'fillarea': False,
    'mode': 'lines',
    'markersymbol': 'dot',
    'markersize': 10}}],
 ['update',
  {'data': [{'x': [1],
     'y': [1],
     'name': 'loss',
     'type': 'scatter',
     'mode': 'lines',
     'textposition': 'right',
     'marker': {'size': 10,
      'symbol': 'dot',
      'line': {'color': '#000000', 'width': 0.5}}}],
   'eid': 'main',
   'layout': {},
   'opts': {'title': 'Training Loss',
    'xlabel': 'Iteration',
    'ylabel': 'Loss',
    'showlegend': False,
    'markers': False,
    'fillarea': False,
    'mode': 'lines',
    'markersymbol': 'dot',
    'markersize': 10},
   'name': 'loss',
   'append': True}]]


def update_fn(engine, batch):
    a = 1
    engine.state.metrics['a'] = a
    return a


def test_visdom():

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

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

    with open(log_filename) as f:
        visdom_log = [json.loads(l.strip()) for l in f.readlines()]

    #
    # Not comparing window name.
    #
    for vl in visdom_log:
        del vl[1]["win"]

    os.remove(log_filename)

    assert visdom_log == expected_log