import time

from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.contrib.handlers.time_profilers import BasicTimeProfiler

from pytest import approx


def _do_nothing_update_fn(engine, batch):
    pass


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

    assert dataflow_results['min/index'][0] == approx(true_dataflow_time_per_ele, abs=1e-1)
    assert dataflow_results['max/index'][0] == approx(true_dataflow_time_per_ele, abs=1e-1)
    assert dataflow_results['mean'] == approx(true_dataflow_time_per_ele, abs=1e-1)
    assert dataflow_results['std'] == approx(0., abs=1e-1)
    assert dataflow_results['total']\
        == approx(true_num_iters * true_dataflow_time_per_ele, abs=1e-1)


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
    processing_results = results['processing_stats']

    assert processing_results['min/index'][0] == approx(true_processing_time, abs=1e-1)
    assert processing_results['max/index'][0] == approx(true_processing_time, abs=1e-1)
    assert processing_results['mean'] == approx(true_processing_time, abs=1e-1)
    assert processing_results['std'] == approx(0., abs=1e-1)
    assert processing_results['total']\
        == approx(true_max_epochs * true_num_iters * true_processing_time, abs=1e-1)


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
    event_results = results['event_handlers_stats']['Events_STARTED']

    assert event_results['min/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['max/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['mean'] == approx(true_event_handler_time, abs=1e-1)


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
    event_results = results['event_handlers_stats']['Events_COMPLETED']

    assert event_results['min/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['max/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['mean'] == approx(true_event_handler_time, abs=1e-1)


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
    event_results = results['event_handlers_stats']['Events_EPOCH_STARTED']

    assert event_results['min/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['max/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['mean'] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['std'] == approx(0., abs=1e-1)
    assert event_results['total'] == approx(
        true_max_epochs * true_event_handler_time, abs=1e-1
    )


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
    event_results = results['event_handlers_stats']['Events_EPOCH_COMPLETED']

    assert event_results['min/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['max/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['mean'] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['std'] == approx(0., abs=1e-1)
    assert event_results['total'] == approx(
        true_max_epochs * true_event_handler_time, abs=1e-1
    )


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
    event_results = results['event_handlers_stats']['Events_ITERATION_STARTED']

    assert event_results['min/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['max/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['mean'] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['std'] == approx(0., abs=1e-1)
    assert event_results['total'] == approx(
        true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1
    )


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
    event_results = results['event_handlers_stats']['Events_ITERATION_COMPLETED']

    assert event_results['min/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['max/index'][0] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['mean'] == approx(true_event_handler_time, abs=1e-1)
    assert event_results['std'] == approx(0., abs=1e-1)
    assert event_results['total'] == approx(
        true_max_epochs * true_num_iters * true_event_handler_time, abs=1e-1
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

    assert event_results['total_time'].item() == approx(
        true_event_handler_time * 6, abs=1e-1)
