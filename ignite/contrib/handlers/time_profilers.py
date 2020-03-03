from collections import OrderedDict

import torch

from ignite.engine import Engine, Events
from ignite.handlers import Timer


class BasicTimeProfiler:
    """
    BasicTimeProfiler can be used to profile the handlers,
    events, data loading and data processing times.

    Examples:

    .. code-block:: python

        #
        # Create an object of the profiler and attach an engine to it
        #
        profiler = BasicTimeProfiler()
        trainer = Engine(train_updater)
        profiler.attach(trainer)
        trainer.run(dataloader, max_epochs=3)

    """

    def __init__(self):
        self._dataflow_timer = Timer()
        self._processing_timer = Timer()
        self._event_handlers_timer = Timer()

        self.dataflow_times = None
        self.processing_times = None
        self.event_handlers_times = None

        self._events = [
            Events.EPOCH_STARTED,
            Events.EPOCH_COMPLETED,
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
            Events.COMPLETED,
        ]
        self._fmethods = [
            self._as_first_epoch_started,
            self._as_first_epoch_completed,
            self._as_first_iter_started,
            self._as_first_iter_completed,
            self._as_first_get_batch_started,
            self._as_first_get_batch_completed,
            self._as_first_completed,
        ]
        self._lmethods = [
            self._as_last_epoch_started,
            self._as_last_epoch_completed,
            self._as_last_iter_started,
            self._as_last_iter_completed,
            self._as_last_get_batch_started,
            self._as_last_get_batch_completed,
            self._as_last_completed,
        ]

    def _reset(self, num_epochs, total_num_iters):
        self.dataflow_times = torch.zeros(total_num_iters)
        self.processing_times = torch.zeros(total_num_iters)
        self.event_handlers_times = {
            Events.STARTED: torch.zeros(1),
            Events.COMPLETED: torch.zeros(1),
            Events.EPOCH_STARTED: torch.zeros(num_epochs),
            Events.EPOCH_COMPLETED: torch.zeros(num_epochs),
            Events.ITERATION_STARTED: torch.zeros(total_num_iters),
            Events.ITERATION_COMPLETED: torch.zeros(total_num_iters),
            Events.GET_BATCH_COMPLETED: torch.zeros(total_num_iters),
            Events.GET_BATCH_STARTED: torch.zeros(total_num_iters),
        }

    def _as_first_started(self, engine):
        if hasattr(engine.state.dataloader, "__len__"):
            num_iters_per_epoch = len(engine.state.dataloader)
        else:
            num_iters_per_epoch = engine.state.epoch_length

        self.max_epochs = engine.state.max_epochs
        self.total_num_iters = self.max_epochs * num_iters_per_epoch
        self._reset(self.max_epochs, self.total_num_iters)

        self.event_handlers_names = {
            e: [
                h.__qualname__ if hasattr(h, "__qualname__") else h.__class__.__name__
                for (h, _, _) in engine._event_handlers[e]
            ]
            for e in Events
            if e != Events.EXCEPTION_RAISED
        }

        # Setup all other handlers:
        engine._event_handlers[Events.STARTED].append((self._as_last_started, (), {}))

        for e, m in zip(self._events, self._fmethods):
            engine._event_handlers[e].insert(0, (m, (), {}))

        for e, m in zip(self._events, self._lmethods):
            engine._event_handlers[e].append((m, (), {}))

        # Let's go
        self._event_handlers_timer.reset()

    def _as_last_started(self, engine):
        self.event_handlers_times[Events.STARTED][0] = self._event_handlers_timer.value()

    def _as_first_epoch_started(self, engine):
        self._event_handlers_timer.reset()

    def _as_last_epoch_started(self, engine):
        t = self._event_handlers_timer.value()
        e = engine.state.epoch - 1
        self.event_handlers_times[Events.EPOCH_STARTED][e] = t

    def _as_first_get_batch_started(self, engine):
        self._event_handlers_timer.reset()
        self._dataflow_timer.reset()

    def _as_last_get_batch_started(self, engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.GET_BATCH_STARTED][i] = t

    def _as_first_get_batch_completed(self, engine):
        self._event_handlers_timer.reset()

    def _as_last_get_batch_completed(self, engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.GET_BATCH_COMPLETED][i] = t

        d = self._dataflow_timer.value()
        self.dataflow_times[i] = d

        self._dataflow_timer.reset()

    def _as_first_iter_started(self, engine):
        self._event_handlers_timer.reset()

    def _as_last_iter_started(self, engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.ITERATION_STARTED][i] = t

        self._processing_timer.reset()

    def _as_first_iter_completed(self, engine):
        t = self._processing_timer.value()
        i = engine.state.iteration - 1
        self.processing_times[i] = t

        self._event_handlers_timer.reset()

    def _as_last_iter_completed(self, engine):
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.ITERATION_COMPLETED][i] = t

    def _as_first_epoch_completed(self, engine):
        self._event_handlers_timer.reset()

    def _as_last_epoch_completed(self, engine):
        t = self._event_handlers_timer.value()
        e = engine.state.epoch - 1
        self.event_handlers_times[Events.EPOCH_COMPLETED][e] = t

    def _as_first_completed(self, engine):
        self._event_handlers_timer.reset()

    def _as_last_completed(self, engine):
        self.event_handlers_times[Events.COMPLETED][0] = self._event_handlers_timer.value()

        # Remove added handlers:
        engine.remove_event_handler(self._as_last_started, Events.STARTED)

        for e, m in zip(self._events, self._fmethods):
            engine.remove_event_handler(m, e)

        for e, m in zip(self._events, self._lmethods):
            engine.remove_event_handler(m, e)

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError("Argument engine should be ignite.engine.Engine, " "but given {}".format(type(engine)))

        if not engine.has_event_handler(self._as_first_started):
            engine._event_handlers[Events.STARTED].insert(0, (self._as_first_started, (), {}))

    @staticmethod
    def _compute_basic_stats(data):
        return OrderedDict(
            [
                ("min/index", (torch.min(data).item(), torch.argmin(data).item())),
                ("max/index", (torch.max(data).item(), torch.argmax(data).item())),
                ("mean", torch.mean(data).item()),
                ("std", torch.std(data).item()),
                ("total", torch.sum(data).item()),
            ]
        )

    def get_results(self):
        """
        Method to fetch the aggregated profiler results after the engine is run

        .. code-block:: python

            results = profiler.get_results()

        """
        events_to_ignore = [Events.EXCEPTION_RAISED]
        total_eh_time = sum([sum(self.event_handlers_times[e]) for e in Events if e not in events_to_ignore])

        return OrderedDict(
            [
                ("processing_stats", self._compute_basic_stats(self.processing_times)),
                ("dataflow_stats", self._compute_basic_stats(self.dataflow_times)),
                (
                    "event_handlers_stats",
                    dict(
                        [
                            (str(e.name).replace(".", "_"), self._compute_basic_stats(self.event_handlers_times[e]))
                            for e in Events
                            if e not in events_to_ignore
                        ]
                        + [("total_time", total_eh_time)]
                    ),
                ),
                (
                    "event_handlers_names",
                    {str(e.name).replace(".", "_") + "_names": v for e, v in self.event_handlers_names.items()},
                ),
            ]
        )

    def write_results(self, output_path):
        """
        Method to store the unaggregated profiling results to a csv file

        .. code-block:: python

            profiler.write_results('path_to_dir/awesome_filename.csv')

        Example output:

        .. code-block:: text

            -----------------------------------------------------------------
            epoch iteration processing_stats dataflow_stats Event_STARTED ...
            1.0     1.0        0.00003         0.252387        0.125676
            1.0     2.0        0.00029         0.252342        0.125123

        """
        try:
            import pandas as pd
        except ImportError:
            print("Need pandas to write results as files")
            return

        iters_per_epoch = self.total_num_iters // self.max_epochs

        epochs = torch.arange(self.max_epochs, dtype=torch.float32).repeat_interleave(iters_per_epoch) + 1
        iterations = torch.arange(self.total_num_iters, dtype=torch.float32) + 1
        processing_stats = self.processing_times
        dataflow_stats = self.dataflow_times

        event_started = self.event_handlers_times[Events.STARTED].repeat_interleave(self.total_num_iters)
        event_completed = self.event_handlers_times[Events.COMPLETED].repeat_interleave(self.total_num_iters)
        event_epoch_started = self.event_handlers_times[Events.EPOCH_STARTED].repeat_interleave(iters_per_epoch)
        event_epoch_completed = self.event_handlers_times[Events.EPOCH_COMPLETED].repeat_interleave(iters_per_epoch)

        event_iter_started = self.event_handlers_times[Events.ITERATION_STARTED]
        event_iter_completed = self.event_handlers_times[Events.ITERATION_COMPLETED]
        event_batch_started = self.event_handlers_times[Events.GET_BATCH_STARTED]
        event_batch_completed = self.event_handlers_times[Events.GET_BATCH_COMPLETED]

        results_dump = torch.stack(
            [
                epochs,
                iterations,
                processing_stats,
                dataflow_stats,
                event_started,
                event_completed,
                event_epoch_started,
                event_epoch_completed,
                event_iter_started,
                event_iter_completed,
                event_batch_started,
                event_batch_completed,
            ],
            dim=1,
        ).numpy()

        results_df = pd.DataFrame(
            data=results_dump,
            columns=[
                "epoch",
                "iteration",
                "processing_stats",
                "dataflow_stats",
                "Event_STARTED",
                "Event_COMPLETED",
                "Event_EPOCH_STARTED",
                "Event_EPOCH_COMPLETED",
                "Event_ITERATION_STARTED",
                "Event_ITERATION_COMPLETED",
                "Event_GET_BATCH_STARTED",
                "Event_GET_BATCH_COMPLETED",
            ],
        )
        results_df.to_csv(output_path, index=False)

    @staticmethod
    def print_results(results):
        """
        Method to print the aggregated results from the profiler

        .. code-block:: python

            profiler.print_results(results)

        Example output:

        .. code-block:: text

            --------------------------------------------
            - Time profiling results:
            --------------------------------------------
            Processing function time stats (in seconds):
                min/index: (2.9754010029137135e-05, 0)
                max/index: (2.9754010029137135e-05, 0)
                mean: 2.9754010029137135e-05
                std: nan
                total: 2.9754010029137135e-05

            Dataflow time stats (in seconds):
                min/index: (0.2523871660232544, 0)
                max/index: (0.2523871660232544, 0)
                mean: 0.2523871660232544
                std: nan
                total: 0.2523871660232544

            Time stats of event handlers (in seconds):
            - Total time spent:
                1.0080009698867798

            - Events.STARTED:
                min/index: (0.1256754994392395, 0)
                max/index: (0.1256754994392395, 0)
                mean: 0.1256754994392395
                std: nan
                total: 0.1256754994392395

            Handlers names:
            ['BasicTimeProfiler._as_first_started', 'delay_start']
            --------------------------------------------
        """

        def odict_to_str(d):
            out = ""
            for k, v in d.items():
                out += "\t{}: {}\n".format(k, v)
            return out

        others = {
            k: odict_to_str(v) if isinstance(v, OrderedDict) else v for k, v in results["event_handlers_stats"].items()
        }

        others.update(results["event_handlers_names"])

        output_message = """
--------------------------------------------
- Time profiling results:
--------------------------------------------

Processing function time stats (in seconds):
{processing_stats}

Dataflow time stats (in seconds):
{dataflow_stats}

Time stats of event handlers (in seconds):
- Total time spent:
\t{total_time}

- Events.STARTED:
{STARTED}
Handlers names:
{STARTED_names}

- Events.EPOCH_STARTED:
{EPOCH_STARTED}
Handlers names:
{EPOCH_STARTED_names}

- Events.ITERATION_STARTED:
{ITERATION_STARTED}
Handlers names:
{ITERATION_STARTED_names}

- Events.ITERATION_COMPLETED:
{ITERATION_COMPLETED}
Handlers names:
{ITERATION_COMPLETED_names}

- Events.EPOCH_COMPLETED:
{EPOCH_COMPLETED}
Handlers names:
{EPOCH_COMPLETED_names}

- Events.COMPLETED:
{COMPLETED}
Handlers names:
{COMPLETED_names}

""".format(
            processing_stats=odict_to_str(results["processing_stats"]),
            dataflow_stats=odict_to_str(results["dataflow_stats"]),
            **others,
        )
        print(output_message)
        return output_message
