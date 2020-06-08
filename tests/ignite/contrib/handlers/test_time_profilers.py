import sys
import os
import time

import pytest
from pytest import approx

from ignite.contrib.handlers.time_profilers import BasicTimeProfiler
from ignite.engine import Engine, Events


def _do_nothing_update_fn(engine, batch):
    pass


def get_prepared_engine(true_event_handler_time):
    dummy_trainer = Engine(_do_nothing_update_fn)

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

    @dummy_trainer.on(Events.GET_BATCH_STARTED)
    def delay_get_batch_started(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.GET_BATCH_COMPLETED)
    def delay_get_batch_completed(engine):
        time.sleep(true_event_handler_time)

    return dummy_trainer


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
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
    dummy_trainer.run(dummy_data_loader(dummy_data), max_epochs=true_max_epochs, epoch_length=true_num_iters)
    results = profiler.get_results()
    dataflow_results = results["dataflow_stats"]

    assert dataflow_results["min/index"][0] == approx(true_dataflow_time_per_ele, abs=1e-1)
    assert dataflow_results["max/index"][0] == approx(true_dataflow_time_per_ele, abs=1e-1)
    assert dataflow_results["mean"] == approx(true_dataflow_time_per_ele, abs=1e-1)
    assert dataflow_results["std"] == approx(0.0, abs=1e-1)
    assert dataflow_results["total"] == approx(true_num_iters * true_dataflow_time_per_ele, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_processing_timer():
    true_processing_time = 0.1
    true_max_epochs = 2
    true_num_iters = 2

    def train_updater(engine, batch):
        time.sleep(true_processing_time)

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(train_updater)
    profiler.attach(dummy_trainer)
    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    processing_results = results["processing_stats"]

    assert processing_results["min/index"][0] == approx(true_processing_time, abs=1e-1)
    assert processing_results["max/index"][0] == approx(true_processing_time, abs=1e-1)
    assert processing_results["mean"] == approx(true_processing_time, abs=1e-1)
    assert processing_results["std"] == approx(0.0, abs=1e-1)
    assert processing_results["total"] == approx(true_max_epochs * true_num_iters * true_processing_time, abs=1e-1)


def test_event_handler_started():
    true_event_handler_time = 0.1
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
    event_results = results["event_handlers_stats"]["STARTED"]

    assert event_results["total"] == approx(true_event_handler_time, abs=1e-1)


def test_event_handler_completed():
    true_event_handler_time = 0.1
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
    event_results = results["event_handlers_stats"]["COMPLETED"]

    assert event_results["total"] == approx(true_event_handler_time, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_event_handler_epoch_started():
    true_event_handler_time = 0.1
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
    event_results = results["event_handlers_stats"]["EPOCH_STARTED"]

    assert event_results["min/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["max/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["mean"] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["std"] == approx(0.0, abs=1e-1)
    assert event_results["total"] == approx(true_max_epochs * true_event_handler_time, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_event_handler_epoch_completed():
    true_event_handler_time = 0.1
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
    event_results = results["event_handlers_stats"]["EPOCH_COMPLETED"]

    assert event_results["min/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["max/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["mean"] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["std"] == approx(0.0, abs=1e-1)
    assert event_results["total"] == approx(true_max_epochs * true_event_handler_time, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_event_handler_iteration_started():
    true_event_handler_time = 0.1
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
    event_results = results["event_handlers_stats"]["ITERATION_STARTED"]

    assert event_results["min/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["max/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["mean"] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["std"] == approx(0.0, abs=1e-1)
    assert event_results["total"] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_event_handler_iteration_completed():
    true_event_handler_time = 0.1
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
    event_results = results["event_handlers_stats"]["ITERATION_COMPLETED"]

    assert event_results["min/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["max/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["mean"] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["std"] == approx(0.0, abs=1e-1)
    assert event_results["total"] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_event_handler_get_batch_started():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.GET_BATCH_STARTED)
    def delay_get_batch_started(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results["event_handlers_stats"]["GET_BATCH_STARTED"]

    assert event_results["min/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["max/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["mean"] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["std"] == approx(0.0, abs=1e-1)
    assert event_results["total"] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_event_handler_get_batch_completed():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.GET_BATCH_COMPLETED)
    def delay_get_batch_completed(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results["event_handlers_stats"]["GET_BATCH_COMPLETED"]

    assert event_results["min/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["max/index"][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["mean"] == approx(true_event_handler_time, abs=1e-1)
    assert event_results["std"] == approx(0.0, abs=1e-1)
    assert event_results["total"] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
def test_event_handler_total_time():
    true_event_handler_time = 0.125
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

    @dummy_trainer.on(Events.GET_BATCH_STARTED)
    def delay_get_batch_started(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.GET_BATCH_COMPLETED)
    def delay_get_batch_completed(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results["event_handlers_stats"]

    assert event_results["total_time"].item() == approx(true_event_handler_time * 8, abs=1e-1)


def test_write_results(dirname):
    true_event_handler_time = 0.125
    true_max_epochs = 3
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = get_prepared_engine(true_event_handler_time)
    profiler.attach(dummy_trainer)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    fp = os.path.join(dirname, "test_log.csv")
    profiler.write_results(fp)

    assert os.path.isfile(fp)

    file_length = 0
    with open(fp) as f:
        for l in f:
            file_length += 1

    assert file_length == (true_max_epochs * true_num_iters) + 1


def test_print_results(capsys):

    true_max_epochs = 1
    true_num_iters = 5

    profiler = BasicTimeProfiler()
    dummy_trainer = get_prepared_engine(true_event_handler_time=0.0125)
    profiler.attach(dummy_trainer)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    BasicTimeProfiler.print_results(profiler.get_results())

    captured = capsys.readouterr()
    out = captured.out
    assert "BasicTimeProfiler._" not in out
    assert "nan" not in out


def test_get_intermediate_results_during_run(capsys):
    true_event_handler_time = 0.0645
    true_max_epochs = 2
    true_num_iters = 5

    profiler = BasicTimeProfiler()
    dummy_trainer = get_prepared_engine(true_event_handler_time)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.ITERATION_COMPLETED(every=3))
    def log_results(_):
        results = profiler.get_results()
        profiler.print_results(results)
        captured = capsys.readouterr()
        out = captured.out
        assert "BasicTimeProfiler._" not in out
        assert "nan" not in out
        assert " min/index: (0.0, " not in out, out

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
