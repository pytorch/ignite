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

        from ignite.contrib.handlers import BasicTimeProfiler

        trainer = Engine(train_updater)

        # Create an object of the profiler and attach an engine to it
        profiler = BasicTimeProfiler()
        profiler.attach(trainer)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_intermediate_results():
            profiler.print_results(profiler.get_results())

        trainer.run(dataloader, max_epochs=3)

        profiler.write_results('path_to_dir/time_profiling.csv')

    """

    events_to_ignore = [
        Events.EXCEPTION_RAISED,
        Events.TERMINATE,
        Events.TERMINATE_SINGLE_EPOCH,
        Events.DATALOADER_STOP_ITERATION,
    ]

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
                if "BasicTimeProfiler." not in repr(h)  # avoid adding internal handlers into output
            ]
            for e in Events
            if e not in self.events_to_ignore
        }

        # Setup all other handlers:
        engine._event_handlers[Events.STARTED].append((self._as_last_started, (engine,), {}))

        for e, m in zip(self._events, self._fmethods):
            engine._event_handlers[e].insert(0, (m, (engine,), {}))

        for e, m in zip(self._events, self._lmethods):
            engine._event_handlers[e].append((m, (engine,), {}))

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
            engine._event_handlers[Events.STARTED].insert(0, (self._as_first_started, (engine,), {}))

    @staticmethod
    def _compute_basic_stats(data):
        # compute on non-zero data:
        data = data[data > 0]
        out = [("total", torch.sum(data).item() if len(data) > 0 else "not yet triggered")]
        if len(data) > 1:
            out += [
                ("min/index", (torch.min(data).item(), torch.argmin(data).item())),
                ("max/index", (torch.max(data).item(), torch.argmax(data).item())),
                ("mean", torch.mean(data).item()),
                ("std", torch.std(data).item()),
            ]
        return OrderedDict(out)

    def get_results(self):
        """
        Method to fetch the aggregated profiler results after the engine is run

        .. code-block:: python

            results = profiler.get_results()

        """
        total_eh_time = sum([(self.event_handlers_times[e]).sum() for e in Events if e not in self.events_to_ignore])

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
                            if e not in self.events_to_ignore
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

             ----------------------------------------------------
            | Time profiling stats (in seconds):                 |
             ----------------------------------------------------
            total  |  min/index  |  max/index  |  mean  |  std

            Processing function:
            157.46292 | 0.01452/1501 | 0.26905/0 | 0.07730 | 0.01258

            Dataflow:
            6.11384 | 0.00008/1935 | 0.28461/1551 | 0.00300 | 0.02693

            Event handlers:
            2.82721

            - Events.STARTED: []
            0.00000

            - Events.EPOCH_STARTED: []
            0.00006 | 0.00000/0 | 0.00000/17 | 0.00000 | 0.00000

            - Events.ITERATION_STARTED: ['PiecewiseLinear']
            0.03482 | 0.00001/188 | 0.00018/679 | 0.00002 | 0.00001

            - Events.ITERATION_COMPLETED: ['TerminateOnNan']
            0.20037 | 0.00006/866 | 0.00089/1943 | 0.00010 | 0.00003

            - Events.EPOCH_COMPLETED: ['empty_cuda_cache', 'training.<locals>.log_elapsed_time', ]
            2.57860 | 0.11529/0 | 0.14977/13 | 0.12893 | 0.00790

            - Events.COMPLETED: []
            not yet triggered

        """

        def to_str(v):
            if isinstance(v, str):
                return v
            elif isinstance(v, tuple):
                return "{:.5f}/{}".format(v[0], v[1])
            return "{:.5f}".format(v)

        def odict_to_str(d):
            out = " | ".join([to_str(v) for v in d.values()])
            return out

        others = {
            k: odict_to_str(v) if isinstance(v, OrderedDict) else v for k, v in results["event_handlers_stats"].items()
        }

        others.update(results["event_handlers_names"])

        output_message = """
 ----------------------------------------------------
| Time profiling stats (in seconds):                 |
 ----------------------------------------------------
total  |  min/index  |  max/index  |  mean  |  std

Processing function:
{processing_stats}

Dataflow:
{dataflow_stats}

Event handlers:
{total_time:.5f}

- Events.STARTED: {STARTED_names}
{STARTED}

- Events.EPOCH_STARTED: {EPOCH_STARTED_names}
{EPOCH_STARTED}

- Events.ITERATION_STARTED: {ITERATION_STARTED_names}
{ITERATION_STARTED}

- Events.ITERATION_COMPLETED: {ITERATION_COMPLETED_names}
{ITERATION_COMPLETED}

- Events.EPOCH_COMPLETED: {EPOCH_COMPLETED_names}
{EPOCH_COMPLETED}

- Events.COMPLETED: {COMPLETED_names}
{COMPLETED}
""".format(
            processing_stats=odict_to_str(results["processing_stats"]),
            dataflow_stats=odict_to_str(results["dataflow_stats"]),
            **others,
        )
        print(output_message)
        return output_message
