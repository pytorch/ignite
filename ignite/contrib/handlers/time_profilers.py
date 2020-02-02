from collections import OrderedDict

import torch

from ignite.engine import Engine, Events
from ignite.handlers import Timer


class BasicTimeProfiler(object):
    """
    BasicTimeProfiler can be used to profile the handlers,
    events, data loading and data processing times.

    Args:
        None

    Attributes:
        data_flow_times (torch.Tensor): tile elapsed during data loading
            for each iteration
        processing_times (torch.Tensor): time elapsed during data processing
            for each iteration
        event_handler_times (dict): time elapsed during execution
            of each event handler attached to the engine

    Example usage:
        Create an object of the profiler and attach an engine to it

        >>> profiler = BasicTimeProfiler()
        >>> trainer = Engine(train_updater)
        >>> profiler.attach(trainer)
        >>> trainer.run(dataloader, max_epochs=3)

        Once the engine is run, we can fetch the profiler stats using,

        >>> results = profiler.get_results()
        >>> profiler.print_results(results)
    """

    def __init__(self):
        self._dataflow_timer = Timer()
        self._processing_timer = Timer()
        self._event_handlers_timer = Timer()

        self._events = [
            Events.EPOCH_STARTED,
            Events.EPOCH_COMPLETED,
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
            Events.COMPLETED
        ]
        self._fmethods = [
            self._as_first_epoch_started,
            self._as_first_epoch_completed,
            self._as_first_iter_started,
            self._as_first_iter_completed,
            self._as_first_get_batch_started,
            self._as_first_get_batch_completed,
            self._as_first_completed
        ]
        self._lmethods = [
            self._as_last_epoch_started,
            self._as_last_epoch_completed,
            self._as_last_iter_started,
            self._as_last_iter_completed,
            self._as_last_get_batch_started,
            self._as_last_get_batch_completed,
            self._as_last_completed
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
            Events.GET_BATCH_STARTED: torch.zeros(total_num_iters)
        }

    def _as_first_started(self, engine):
        # breakpoint()
        if hasattr(engine.state.dataloader, "__len__"):
            num_iters_per_epoch = len(engine.state.dataloader)
        else:
            num_iters_per_epoch = engine.state.epoch_length

        num_iters = engine.state.max_epochs * num_iters_per_epoch
        self._reset(engine.state.max_epochs, num_iters)

        self.event_handlers_names = {
            e: [h.__qualname__ if hasattr(h, "__qualname__") else h.__class__.__name__
                for (h, _, _) in engine._event_handlers[e]]
            for e in Events if e != Events.EXCEPTION_RAISED
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
        # breakpoint()
        self.event_handlers_times[Events.STARTED][0] = self._event_handlers_timer.value()

    def _as_first_epoch_started(self, engine):
        # breakpoint()
        self._event_handlers_timer.reset()

    def _as_last_epoch_started(self, engine):
        # breakpoint()
        t = self._event_handlers_timer.value()
        e = engine.state.epoch - 1
        self.event_handlers_times[Events.EPOCH_STARTED][e] = t

        self._dataflow_timer.reset()

    def _as_first_get_batch_started(self, engine):
        # breakpoint()
        self._event_handlers_timer.reset()

    def _as_last_get_batch_started(self, engine):
        # breakpoint()
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.GET_BATCH_STARTED][i] = t

    def _as_first_get_batch_completed(self, engine):
        # breakpoint()
        self._event_handlers_timer.reset()

    def _as_last_get_batch_completed(self, engine):
        # breakpoint()
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.GET_BATCH_COMPLETED][i] = t

    def _as_first_iter_started(self, engine):
        # breakpoint()
        t = self._dataflow_timer.value()
        i = engine.state.iteration - 1
        self.dataflow_times[i] = t

        self._event_handlers_timer.reset()

    def _as_last_iter_started(self, engine):
        # breakpoint()
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.ITERATION_STARTED][i] = t

        self._processing_timer.reset()

    def _as_first_iter_completed(self, engine):
        # breakpoint()
        t = self._processing_timer.value()
        i = engine.state.iteration - 1
        self.processing_times[i] = t

        self._event_handlers_timer.reset()

    def _as_last_iter_completed(self, engine):
        # breakpoint()
        t = self._event_handlers_timer.value()
        i = engine.state.iteration - 1
        self.event_handlers_times[Events.ITERATION_COMPLETED][i] = t

        self._dataflow_timer.reset()

    def _as_first_epoch_completed(self, engine):
        # breakpoint()
        self._event_handlers_timer.reset()

    def _as_last_epoch_completed(self, engine):
        # breakpoint()
        t = self._event_handlers_timer.value()
        e = engine.state.epoch - 1
        self.event_handlers_times[Events.EPOCH_COMPLETED][e] = t

    def _as_first_completed(self, engine):
        # breakpoint()
        self._event_handlers_timer.reset()

    def _as_last_completed(self, engine):
        # breakpoint()
        self.event_handlers_times[Events.COMPLETED][0] = self._event_handlers_timer.value()

        # Remove added handlers:
        engine.remove_event_handler(self._as_last_started, Events.STARTED)

        for e, m in zip(self._events, self._fmethods):
            engine.remove_event_handler(m, e)

        for e, m in zip(self._events, self._lmethods):
            engine.remove_event_handler(m, e)

    def attach(self, engine):
        if not isinstance(engine, Engine):
            raise TypeError("Argument engine should be ignite.engine.Engine, "
                            "but given {}".format(type(engine)))

        if not engine.has_event_handler(self._as_first_started):
            engine._event_handlers[Events.STARTED]\
                .insert(0, (self._as_first_started, (), {}))

    @staticmethod
    def _compute_basic_stats(data):
        return OrderedDict([
            ('min/index', (torch.min(data).item(), torch.argmin(data).item())),
            ('max/index', (torch.max(data).item(), torch.argmax(data).item())),
            ('mean', torch.mean(data).item()),
            ('std', torch.std(data).item()),
            ('total', torch.sum(data).item())
        ])

    def get_results(self):
        events_to_ignore = [
            Events.EXCEPTION_RAISED,
            # Events.GET_BATCH_COMPLETED,
            # Events.GET_BATCH_STARTED
        ]
        total_eh_time = sum([sum(self.event_handlers_times[e]) for e in Events if e not in events_to_ignore])
        return OrderedDict([
            ("processing_stats", self._compute_basic_stats(self.processing_times)),
            ("dataflow_stats", self._compute_basic_stats(self.dataflow_times)),
            ("event_handlers_stats",
                dict([(str(e).replace(".", "_"), self._compute_basic_stats(self.event_handlers_times[e]))
                      for e in Events if e not in events_to_ignore] + [("total_time", total_eh_time)])
             ),
            ("event_handlers_names", {str(e).replace(".", "_") + "_names": v
                                      for e, v in self.event_handlers_names.items()})
        ])

    @staticmethod
    def print_results(results):

        def odict_to_str(d):
            out = ""
            for k, v in d.items():
                out += "\t{}: {}\n".format(k, v)
            return out

        others = {k: odict_to_str(v) if isinstance(v, OrderedDict) else v
                  for k, v in results['event_handlers_stats'].items()}

        others.update(results['event_handlers_names'])

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
{Events_STARTED}
Handlers names:
{Events_STARTED_names}

- Events.EPOCH_STARTED:
{Events_EPOCH_STARTED}
Handlers names:
{Events_EPOCH_STARTED_names}

- Events.ITERATION_STARTED:
{Events_ITERATION_STARTED}
Handlers names:
{Events_ITERATION_STARTED_names}

- Events.ITERATION_COMPLETED:
{Events_ITERATION_COMPLETED}
Handlers names:
{Events_ITERATION_COMPLETED_names}

- Events.EPOCH_COMPLETED:
{Events_EPOCH_COMPLETED}
Handlers names:
{Events_EPOCH_COMPLETED_names}

- Events.COMPLETED:
{Events_COMPLETED}
Handlers names:
{Events_COMPLETED_names}

""".format(processing_stats=odict_to_str(results['processing_stats']),
           dataflow_stats=odict_to_str(results['dataflow_stats']),
           **others)
        print(output_message)
        return output_message

    @staticmethod
    def write_results(output_path):
        try:
            import pandas as pd
        except ImportError:
            print("Need pandas to write results as files")
            return

        raise NotImplementedError("")
