import torch
import contextlib
from typing import Callable
from ignite.engine import Events, Engine


class ExecutionStats(object):
    # Abstracted from
    # https://github.com/pytorch/pytorch/blob/v1.2.0/torch/utils/throughput_benchmark.py#L27
    # for linting support
    def __init__(self):
        ...

    @property
    def latency_avg_ms(self) -> float:
        ...

    @property
    def num_iters(self) -> int:
        ...

    @property
    def iters_per_second(self) -> float:
        ...

    @property
    def total_time_seconds(self) -> float:
        ...


# TODO: Discuss device implications
class ThroughputBenchmarkWrapper:
    """
    This class is wrapper around the ThroughputBenchmark wrapper from torch.
    ThroughputBenchmark is responsible for executing a PyTorch
    module (nn.Module or ScriptModule)
    under an inference server like load. It can emulate multiple calling threads
    to a single module provided.

    Please note that even though nn.Module is supported, it might incur an overhead
    from the need to hold GIL every time we execute Python code or pass around
    inputs as Python objects. As soon as you have a ScriptModule version of your
    model for inference deployment it is better to switch to using it in this
    benchmark.

    Args:
        model (torch.nn.Module): model which will
            be used to run the benchmark on. Please keep in mind that most reliable
            test results will be achieved with a `ScriptModule`.
        num_calling_threads (int, optional): Number of threads that will call the model.
            Defaults to 1.
        num_warmup_iters (int, optional): Warmup iters are used to make sure we run a module
            a few times before actually measuring things. This way we avoid cold
            caches and any other similar problems. This is the number of warmup
            iterations for each of the threads in separate.
            Defaults to 10.
        num_iters (int, optional): Number of iterations the benchmark should run with.
            This number is separate from the warmup iterations. Also the number is
            shared across all the threads. Once the num_iters iterations across all
            the threads is reached, we will stop execution. Though total number of
            iterations might be slightly larger. Which is reported as
            stats.num_iters where stats is the result of this function.
            Defaults to 100.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers import ThroughputBenchmarkWrapper

            # dataloader gets batches of (X, y) with many samples per batch
            dataloader = ...
            evaluator = ...
            model = ...
            throughput_benchmark = ThroughputBenchmarkWrapper(model)

            # We decide that 10 batches of the datasampler are enough different samples for benchmark
            # Engine will be stopped after these 10 batches!
            max_batches = 10
            # batch are of type (X, y) -> default input_transform is can be used
            with throughput_benchmark.attach(evaluator, max_batches=max_batches) as evaluator_with_benchmark:
                evaluator_with_benchmark.run(dataloader)
            # get the results
            stats = throughput_benchmark.stats
            # print the results
            print(stats)

    Note:
        Please, also keep in mind that all other handlers attached the trainer will be executed during benchmark run.

    References:

        pytorch/ThroughputBenchmark: https://github.com/pytorch/pytorch/blob/v1.2.0/torch/utils/throughput_benchmark.py
    """

    def __init__(
        self, model: torch.nn.Module, num_calling_threads: int = 1, num_warmup_iters: int = 10, num_iters: int = 100,
    ):
        try:
            from torch.utils import ThroughputBenchmark
        except ImportError:
            raise RuntimeError("This method requires at least pytorch version  1.2.0 to be installed")

        # These error messages are a bit more clear than the ones from
        # the C++ file in my opinion.
        if not isinstance(num_calling_threads, int):
            raise TypeError(
                "Number of calling threads should be int type, but is: {}".format(type(num_calling_threads))
            )
        if not isinstance(num_warmup_iters, int):
            raise TypeError("Number of warmup iterations should be int type, but is: {}".format(type(num_warmup_iters)))
        if not isinstance(num_iters, int):
            raise TypeError("Number of iterations should be int type, but is: {}".format(type(num_iters)))
        if num_calling_threads < 1:
            raise ValueError("Number of calling threads has to be larger than 0! Given: {}".format(num_calling_threads))
        if num_warmup_iters < 0:
            raise ValueError("Warmup iterations has to be larger or equal to 0. Given {}".format(num_warmup_iters))
        if num_iters < 1:
            raise ValueError("Number of iterations has to be larger than 0! Given {}".format(num_iters))
        self._bench = ThroughputBenchmark(model)
        self._num_calling_threads = num_calling_threads
        self._num_warmup_iters = num_warmup_iters
        self._num_iters = num_iters
        self._stats = None

    def _batch_logger(self, engine: Engine, input_transform: Callable) -> None:
        """
        Load batch from the `engine.state.bach`, process with
        `input_transform` and add input data to benchmark.

        Args:
            engine (Engine): Engine from which state batch will be loaded
            input_transform (Callable): Transform function used on batch
                before adding as an input example to the benchmark.
        """
        input_data = input_transform(engine.state.batch)
        self._bench.add_input(input_data)

    def _run(self, _: Engine) -> None:
        """
        Run actual benchmark from `torch.utils`.
        Will be called over event handler and will give
        current engine as argument, which won't be used here.

        Args:
            engine (Engine): Given from event handler (not used)
        """
        self._stats = self._bench.benchmark(
            num_calling_threads=self._num_calling_threads,
            num_warmup_iters=self._num_warmup_iters,
            num_iters=self._num_iters,
        )

    def _detach(self, engine: Engine) -> None:
        """
        Removes event handler :py:meth:`~._batch_logger` and
        :py:meth:`~._run` from the engine.

        Args:
            engine (Engine): Engine from which events handlers will be removed
        """
        if engine.has_event_handler(self._batch_logger, Events.ITERATION_STARTED):
            engine.remove_event_handler(self._batch_logger, Events.ITERATION_STARTED)
        if engine.has_event_handler(self._run, Events.COMPLETED):
            engine.remove_event_handler(self._run, Events.COMPLETED)

    @contextlib.contextmanager
    def attach(
        self, engine: Engine, max_batches: int = 10, input_transform: Callable = lambda input_batch: input_batch[0]
    ):
        """
        Attaches throughput_benchmark to a given engine.
        As for a benchmark not the complete dataset is required, and could consume too much memory,
        only the first `max_batches` are used for the benchmark.
        `input_transform` is used to retrieve the corresponding input for the forward function
        of the model from the batch.

        Args:
            engine (Engine): throughput_benchmark is attached to this engine. Please, keep in mind that all
                attached handlers will be executed.
            max_batches (int, optional): Number of batches to use for the benchmark. After sampling
                this many batches, the engine will terminate early if there are remaining iterations.
                To sample complete dataset use: `len(dataloader)`.
                Defaults to 10.
            input_transform (Callable, optional): function that transform the batch from the dataloader
                to the input required for the model input. Defaults to lambda input_batch:input_batch[0].

        Note:
            The batches are processed from the `engine.state`. If the input data is different from the
            data from the dataloader, the wrong data will be benchmarked. Either provide a dataloader
            with all of the transformations, or include the transformations in `input_transform`.

        Returns:
            Engine: engine_with_throughput_benchmark: engine used for testing the throughput

        Example:

            .. code-block:: python

                with throughput_bechmark.attach(engine, max_batches=5) as engine_with_throughput_benchmark:
                    engine_with_throughput_benchmark.run(dataloader)

        """

        def stop_after_max_batches(engine: Engine, event: Events):
            # Events start with 1
            if event <= max_batches:
                if event == max_batches:
                    engine.terminate()
                return True
            return False

        if not isinstance(max_batches, int):
            raise TypeError("Max batches has to be int. Given: {}".format(type(max_batches)))
        if not callable(input_transform):
            raise TypeError("Input transform has to be Callable. Given: {}".format(type(input_transform)))
        if max_batches < 1:
            raise ValueError("Max batches to sample from has to be larger than 0. Given: {}".format(max_batches))

        if not engine.has_event_handler(self._run):
            engine.add_event_handler(
                Events.ITERATION_STARTED(event_filter=stop_after_max_batches), self._batch_logger, input_transform,
            )
            engine.add_event_handler(
                Events.COMPLETED, self._run,
            )

        yield engine
        self._detach(engine)

    @property
    def stats(self) -> ExecutionStats:
        """
        If benchmark was attached and run, the
        execution stats results are returned.

        Returns:
            ExecutionStats:
                Wraps all the important stats int
                a single object, which includes a nice print output.
        """
        if self._stats is None:
            raise RuntimeError("Benchmark wrapper hasn't run yet so results can't be retrieved.")
        return self._stats
