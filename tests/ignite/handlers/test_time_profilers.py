import sys
import time
from unittest.mock import patch

import pytest
from pytest import approx

from ignite.engine import Engine, EventEnum, Events
from ignite.handlers.time_profilers import BasicTimeProfiler, HandlersTimeProfiler

if sys.platform.startswith("darwin"):
    pytest.skip("Skip if on MacOS", allow_module_level=True)


def _do_nothing_update_fn(engine, batch):
    pass


def get_prepared_engine_for_basic_profiler(true_event_handler_time):
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


def get_prepared_engine_for_handlers_profiler(true_event_handler_time):
    HANDLERS_SLEEP_COUNT = 11
    PROCESSING_SLEEP_COUNT = 3

    class CustomEvents(EventEnum):
        CUSTOM_STARTED = "custom_started"
        CUSTOM_COMPLETED = "custom_completed"

    def dummy_train_step(engine, batch):
        engine.fire_event(CustomEvents.CUSTOM_STARTED)
        time.sleep(true_event_handler_time)
        engine.fire_event(CustomEvents.CUSTOM_COMPLETED)

    dummy_trainer = Engine(dummy_train_step)
    dummy_trainer.register_events(*CustomEvents)

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

    @dummy_trainer.on(CustomEvents.CUSTOM_STARTED)
    def delay_custom_started(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(CustomEvents.CUSTOM_COMPLETED)
    def delay_custom_completed(engine):
        time.sleep(true_event_handler_time)

    @dummy_trainer.on(Events.EPOCH_STARTED(once=1))
    def do_something_once_on_1_epoch():
        time.sleep(true_event_handler_time)

    return dummy_trainer, HANDLERS_SLEEP_COUNT, PROCESSING_SLEEP_COUNT


def test_profilers_wrong_inputs():
    profiler = BasicTimeProfiler()
    with pytest.raises(TypeError, match=r"Argument engine should be ignite.engine.Engine"):
        profiler.attach(None)

    with pytest.raises(ModuleNotFoundError, match=r"Need pandas to write results as files"):
        with patch.dict("sys.modules", {"pandas": None}):
            profiler.write_results("")

    profiler = HandlersTimeProfiler()
    with pytest.raises(TypeError, match=r"Argument engine should be ignite.engine.Engine"):
        profiler.attach(None)

    with pytest.raises(ModuleNotFoundError, match=r"Need pandas to write results as files"):
        with patch.dict("sys.modules", {"pandas": None}):
            profiler.write_results("")


def test_dataflow_timer_basic_profiler():
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


def test_dataflow_timer_handlers_profiler():
    true_dataflow_time_per_ele = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    def dummy_data_loader(data):
        while True:
            for d in data:
                time.sleep(true_dataflow_time_per_ele)
                yield d

    dummy_data = range(true_num_iters)

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)
    dummy_trainer.run(dummy_data_loader(dummy_data), max_epochs=true_max_epochs, epoch_length=true_num_iters)
    results = profiler.get_results()
    dataflow_results = results[-1]

    assert dataflow_results[0] == "Dataflow"
    # event name
    assert dataflow_results[1] == "None"
    # total
    assert dataflow_results[2] == approx(true_num_iters * true_dataflow_time_per_ele, abs=1e-1)
    # min
    assert dataflow_results[3][0] == approx(true_dataflow_time_per_ele, abs=1e-1)
    # max
    assert dataflow_results[4][0] == approx(true_dataflow_time_per_ele, abs=1e-1)
    # mean
    assert dataflow_results[5] == approx(true_dataflow_time_per_ele, abs=1e-1)
    # stddev
    assert dataflow_results[6] == approx(0.0, abs=1e-1)


def test_processing_timer_basic_profiler():
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


def test_processing_timer_handlers_profiler():
    true_processing_time = 0.1
    true_max_epochs = 2
    true_num_iters = 2

    def train_updater(engine, batch):
        time.sleep(true_processing_time)

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(train_updater)
    profiler.attach(dummy_trainer)
    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    processing_results = results[-2]

    assert processing_results[0] == "Processing"
    # event name
    assert processing_results[1] == "None"
    # total
    assert processing_results[2] == approx(true_max_epochs * true_num_iters * true_processing_time, abs=1e-1)
    # min
    assert processing_results[3][0] == approx(true_processing_time, abs=1e-1)
    # max
    assert processing_results[4][0] == approx(true_processing_time, abs=1e-1)
    # mean
    assert processing_results[5] == approx(true_processing_time, abs=1e-1)
    # stddev
    assert processing_results[6] == approx(0.0, abs=1e-1)


def test_event_handler_started_basic_profiler():
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


def test_event_handler_started_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 2
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.STARTED)
    def delay_start(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]

    assert "delay_start" in event_results[0]
    assert event_results[1] == "STARTED"

    assert event_results[2] == approx(true_event_handler_time, abs=1e-1)  # total


def test_event_handler_completed_basic_profiler():
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


def test_event_handler_completed_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 2
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.COMPLETED)
    def delay_complete(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "delay_complete" in event_results[0]
    assert event_results[1] == "COMPLETED"

    assert event_results[2] == approx(true_event_handler_time, abs=1e-1)  # total


def test_event_handler_epoch_started_basic_profiler():
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


def test_event_handler_epoch_started_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 2
    true_num_iters = 1

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.EPOCH_STARTED)
    def delay_epoch_start(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "delay_epoch_start" in event_results[0]
    assert event_results[1] == "EPOCH_STARTED"

    assert event_results[2] == approx(true_max_epochs * true_event_handler_time, abs=1e-1)  # total
    assert event_results[3][0] == approx(true_event_handler_time, abs=1e-1)  # min
    assert event_results[4][0] == approx(true_event_handler_time, abs=1e-1)  # max
    assert event_results[5] == approx(true_event_handler_time, abs=1e-1)  # mean
    assert event_results[6] == approx(0.0, abs=1e-1)  # stddev


def test_event_handler_epoch_completed_basic_profiler():
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


def test_event_handler_epoch_completed_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 2
    true_num_iters = 1

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.EPOCH_COMPLETED)
    def delay_epoch_complete(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "delay_epoch_complete" in event_results[0]
    assert event_results[1] == "EPOCH_COMPLETED"

    assert event_results[2] == approx(true_max_epochs * true_event_handler_time, abs=1e-1)  # total
    assert event_results[3][0] == approx(true_event_handler_time, abs=1e-1)  # min
    assert event_results[4][0] == approx(true_event_handler_time, abs=1e-1)  # max
    assert event_results[5] == approx(true_event_handler_time, abs=1e-1)  # mean
    assert event_results[6] == approx(0.0, abs=1e-1)  # stddev


def test_event_handler_iteration_started_basic_profiler():
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


def test_event_handler_iteration_started_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.ITERATION_STARTED)
    def delay_iter_start(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "delay_iter_start" in event_results[0]
    assert event_results[1] == "ITERATION_STARTED"

    assert event_results[2] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)  # total
    assert event_results[3][0] == approx(true_event_handler_time, abs=1e-1)  # min
    assert event_results[4][0] == approx(true_event_handler_time, abs=1e-1)  # max
    assert event_results[5] == approx(true_event_handler_time, abs=1e-1)  # mean
    assert event_results[6] == approx(0.0, abs=1e-1)  # stddev


def test_event_handler_iteration_completed_basic_profiler():
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


def test_event_handler_iteration_completed_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.ITERATION_COMPLETED)
    def delay_iter_complete(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "delay_iter_complete" in event_results[0]
    assert event_results[1] == "ITERATION_COMPLETED"

    assert event_results[2] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)  # total
    assert event_results[3][0] == approx(true_event_handler_time, abs=1e-1)  # min
    assert event_results[4][0] == approx(true_event_handler_time, abs=1e-1)  # max
    assert event_results[5] == approx(true_event_handler_time, abs=1e-1)  # mean
    assert event_results[6] == approx(0.0, abs=1e-1)  # stddev


def test_event_handler_get_batch_started_basic_profiler():
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


def test_event_handler_get_batch_started_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.GET_BATCH_STARTED)
    def delay_get_batch_started(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "delay_get_batch_started" in event_results[0]
    assert event_results[1] == "GET_BATCH_STARTED"

    assert event_results[2] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)  # total
    assert event_results[3][0] == approx(true_event_handler_time, abs=1e-1)  # min
    assert event_results[4][0] == approx(true_event_handler_time, abs=1e-1)  # max
    assert event_results[5] == approx(true_event_handler_time, abs=1e-1)  # mean
    assert event_results[6] == approx(0.0, abs=1e-1)  # stddev


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


def test_event_handler_get_batch_completed_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.GET_BATCH_COMPLETED)
    def delay_get_batch_completed(engine):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "delay_get_batch_completed" in event_results[0]
    assert event_results[1] == "GET_BATCH_COMPLETED"

    assert event_results[2] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)  # total
    assert event_results[3][0] == approx(true_event_handler_time, abs=1e-1)  # min
    assert event_results[4][0] == approx(true_event_handler_time, abs=1e-1)  # max
    assert event_results[5] == approx(true_event_handler_time, abs=1e-1)  # mean
    assert event_results[6] == approx(0.0, abs=1e-1)  # stddev


def test_neg_event_filter_threshold_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 1

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.EPOCH_STARTED(once=2))
    def do_something_once_on_2_epoch():
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "do_something_once_on_2_epoch" in event_results[0]
    assert event_results[1] == "EPOCH_STARTED"
    assert event_results[2] == "not triggered"


def test_pos_event_filter_threshold_handlers_profiler():
    true_event_handler_time = HandlersTimeProfiler.EVENT_FILTER_THESHOLD_TIME
    true_max_epochs = 2
    true_num_iters = 1

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.EPOCH_STARTED(once=2))
    def do_something_once_on_2_epoch():
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = results[0]
    assert "do_something_once_on_2_epoch" in event_results[0]
    assert event_results[1] == "EPOCH_STARTED"
    assert event_results[2] == approx(
        (true_max_epochs * true_num_iters * true_event_handler_time) / 2, abs=1e-1
    )  # total


def test_custom_event_with_arg_handlers_profiler():
    true_event_handler_time = 0.1
    true_max_epochs = 1
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer = Engine(_do_nothing_update_fn)
    dummy_trainer.register_events("custom_event")
    profiler.attach(dummy_trainer)

    @dummy_trainer.on(Events.ITERATION_COMPLETED(every=1))
    def trigger_custom_event():
        dummy_trainer.fire_event("custom_event")

    args = [122, 324]

    @dummy_trainer.on("custom_event", args)
    def on_custom_event(args):
        time.sleep(true_event_handler_time)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    event_results = None
    for row in results:
        if row[1] == "custom_event":
            event_results = row
            break
    assert event_results is not None
    assert "on_custom_event" in event_results[0]

    assert event_results[2] == approx(true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1)  # total
    assert event_results[3][0] == approx(true_event_handler_time, abs=1e-1)  # min
    assert event_results[4][0] == approx(true_event_handler_time, abs=1e-1)  # max
    assert event_results[5] == approx(true_event_handler_time, abs=1e-1)  # mean
    assert event_results[6] == approx(0.0, abs=1e-1)  # stddev


def test_event_handler_total_time_basic_profiler():
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


def test_event_handler_total_time_handlers_profiler():
    true_event_handler_time = 0.125
    true_max_epochs = 1
    true_num_iters = 1

    profiler = HandlersTimeProfiler()
    dummy_trainer, handlers_sleep_count, processing_sleep_count = get_prepared_engine_for_handlers_profiler(
        true_event_handler_time
    )
    profiler.attach(dummy_trainer)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    results = profiler.get_results()
    total_handler_stats = results[-3]  # total result row
    total_processing_stats = results[-2]  # processing result row

    assert total_handler_stats[2] == approx(true_event_handler_time * handlers_sleep_count, abs=1e-1)  # total time
    assert total_processing_stats[2] == approx(true_event_handler_time * processing_sleep_count, abs=1e-1)  # total time


def test_write_results_basic_profiler(dirname):
    true_event_handler_time = 0.125
    true_max_epochs = 3
    true_num_iters = 2

    profiler = BasicTimeProfiler()
    dummy_trainer = get_prepared_engine_for_basic_profiler(true_event_handler_time)
    profiler.attach(dummy_trainer)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    fp = dirname / "test_log.csv"
    profiler.write_results(fp)

    assert fp.is_file()

    file_length = 0
    with open(fp) as f:
        for _ in f:
            file_length += 1

    assert file_length == (true_max_epochs * true_num_iters) + 1


def test_write_results_handlers_profiler(dirname):
    true_event_handler_time = 0.125
    true_max_epochs = 3
    true_num_iters = 2

    profiler = HandlersTimeProfiler()
    dummy_trainer, _, _ = get_prepared_engine_for_handlers_profiler(true_event_handler_time)
    profiler.attach(dummy_trainer)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    fp = dirname / "test_log.csv"
    profiler.write_results(fp)

    assert fp.is_file()

    file_length = 0
    with open(fp) as f:
        for _ in f:
            file_length += 1

    assert file_length == (true_max_epochs * true_num_iters) + 1


def test_print_results_basic_profiler(capsys):
    true_max_epochs = 1
    true_num_iters = 5

    profiler = BasicTimeProfiler()
    dummy_trainer = get_prepared_engine_for_basic_profiler(true_event_handler_time=0.0125)
    profiler.attach(dummy_trainer)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    BasicTimeProfiler.print_results(profiler.get_results())

    captured = capsys.readouterr()
    out = captured.out
    assert "BasicTimeProfiler._" not in out
    assert "nan" not in out


def test_print_results_handlers_profiler_handlers_profiler(capsys):
    true_max_epochs = 1
    true_num_iters = 5

    profiler = HandlersTimeProfiler()
    dummy_trainer, _, _ = get_prepared_engine_for_handlers_profiler(true_event_handler_time=0.0125)
    profiler.attach(dummy_trainer)

    dummy_trainer.run(range(true_num_iters), max_epochs=true_max_epochs)
    HandlersTimeProfiler.print_results(profiler.get_results())

    captured = capsys.readouterr()
    out = captured.out
    assert "HandlersTimeProfiler." not in out
    assert "Timer." not in out


def test_get_intermediate_results_during_run_basic_profiler(capsys):
    true_event_handler_time = 0.0645
    true_max_epochs = 2
    true_num_iters = 5

    profiler = BasicTimeProfiler()
    dummy_trainer = get_prepared_engine_for_basic_profiler(true_event_handler_time)
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
