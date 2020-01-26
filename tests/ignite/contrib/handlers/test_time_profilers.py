import os
import warnings
import time

import torch
import torch.nn as nn

from ignite.engine import Engine, Events, State
from ignite.handlers import Timer
from ignite.contrib.handlers.time_profilers import BasicTimeProfiler

import pytest


def _do_nothing_update_fn(engine, batch):
    pass


def _equal(lhs, rhs, round_to=1):
    return round(lhs, round_to) == round(rhs, round_to)


def test_dataflow_timer():
    true_dataflow_time_per_ele = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    def dummy_data_loader(data):
        while True:
            for d in data:
                time.sleep(true_dataflow_time_per_ele)
                yield d

    dummy_data = range(true_num_iters)

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)
    dummy_trainer.run(
        dummy_data_loader(dummy_data),
        max_epochs=true_max_epochs,
        epoch_length=true_num_iters
    )
    results = profiler.get_results()
    dataflow_results = results['dataflow_stats']

    assert _equal(dataflow_results['min/index'][0], true_dataflow_time_per_ele)
    assert _equal(dataflow_results['max/index'][0], true_dataflow_time_per_ele)
    assert _equal(dataflow_results['mean'], true_dataflow_time_per_ele)
    assert _equal(dataflow_results['std'], 0)
    assert _equal(
        dataflow_results['total'],
        true_num_iters * true_dataflow_time_per_ele
    )


def test_processing_timer():
    true_processing_time = 1
    true_max_epochs = 2
    true_num_iters = 2

    def train_updater(engine, batch):
        time.sleep(true_processing_time)

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(train_updater)
    profiler.attach(dummy_trainer)
    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    processing_results = results['processing_stats']

    assert _equal(processing_results['min/index'][0], true_processing_time)
    assert _equal(processing_results['max/index'][0], true_processing_time)
    assert _equal(processing_results['mean'], true_processing_time)
    assert _equal(processing_results['std'], 0)
    assert _equal(
        processing_results['total'],
        true_max_epochs * true_num_iters * true_processing_time
    )


def test_event_handler_started():
    true_event_handler_time = 1
    true_max_epochs = 2
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.STARTED)
    def delay_start(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results['event_handlers_stats']['Events_STARTED']

    assert _equal(event_results['min/index'][0], true_event_handler_time)
    assert _equal(event_results['max/index'][0], true_event_handler_time)
    assert _equal(event_results['mean'], true_event_handler_time)


def test_event_handler_completed():
    true_event_handler_time = 1
    true_max_epochs = 2
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.COMPLETED)
    def delay_complete(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results['event_handlers_stats']['Events_COMPLETED']

    assert _equal(event_results['min/index'][0], true_event_handler_time)
    assert _equal(event_results['max/index'][0], true_event_handler_time)
    assert _equal(event_results['mean'], true_event_handler_time)


def test_event_handler_epoch_started():
    true_event_handler_time = 1
    true_max_epochs = 2
    true_num_iters = 1

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.EPOCH_STARTED)
    def delay_epoch_start(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results['event_handlers_stats']['Events_EPOCH_STARTED']

    assert _equal(event_results['min/index'][0], true_event_handler_time)
    assert _equal(event_results['max/index'][0], true_event_handler_time)
    assert _equal(event_results['mean'], true_event_handler_time)
    assert _equal(event_results['std'], 0)
    assert _equal(
        event_results['total'],
        true_max_epochs * true_event_handler_time
    )


def test_event_handler_epoch_completed():
    true_event_handler_time = 1
    true_max_epochs = 2
    true_num_iters = 1

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.EPOCH_COMPLETED)
    def delay_epoch_complete(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results['event_handlers_stats']['Events_EPOCH_COMPLETED']

    assert _equal(event_results['min/index'][0], true_event_handler_time)
    assert _equal(event_results['max/index'][0], true_event_handler_time)
    assert _equal(event_results['mean'], true_event_handler_time)
    assert _equal(event_results['std'], 0)
    assert _equal(
        event_results['total'],
        true_max_epochs * true_event_handler_time
    )


def test_event_handler_iteration_started():
    true_event_handler_time = 1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.ITERATION_STARTED)
    def delay_iter_start(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results['event_handlers_stats']['Events_ITERATION_STARTED']

    assert _equal(event_results['min/index'][0], true_event_handler_time)
    assert _equal(event_results['max/index'][0], true_event_handler_time)
    assert _equal(event_results['mean'], true_event_handler_time)
    assert _equal(event_results['std'], 0)
    assert _equal(
        event_results['total'],
        true_max_epochs * true_num_iters * true_event_handler_time
    )


def test_event_handler_iteration_completed():
    true_event_handler_time = 1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.ITERATION_COMPLETED)
    def delay_iter_complete(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results['event_handlers_stats']['Events_ITERATION_COMPLETED']

    assert _equal(event_results['min/index'][0], true_event_handler_time)
    assert _equal(event_results['max/index'][0], true_event_handler_time)
    assert _equal(event_results['mean'], true_event_handler_time)
    assert _equal(event_results['std'], 0)
    assert _equal(
        event_results['total'],
        true_max_epochs * true_num_iters * true_event_handler_time
    )


def test_event_handler_total_time():
    true_event_handler_time = 0.5
    true_max_epochs = 1
    true_num_iters = 1

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.STARTED)
    def delay_start(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.COMPLETED)
    def delay_complete(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.EPOCH_STARTED)
    def delay_epoch_start(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.EPOCH_COMPLETED)
    def delay_epoch_complete(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.ITERATION_STARTED)
    def delay_iter_start(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.ITERATION_COMPLETED)
    def delay_iter_complete(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results['event_handlers_stats']

    assert _equal(event_results['total_time'].item(), true_event_handler_time * 6)
