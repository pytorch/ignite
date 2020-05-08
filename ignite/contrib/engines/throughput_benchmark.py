from typing import Callable, Optional, Sequence, Union

import torch

from ignite.engine import Events, Engine
from ignite.utils import convert_tensor


def _prepare_input(
    batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
):
    """
    Prepare batch for adding to benchmark: pass to a device with options.
    """
    x, _ = batch
    return convert_tensor(x, device=device, non_blocking=non_blocking)


class BenchmarkEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            from torch.utils import ThroughputBenchmark
            from torch.utils.throughput_benchmark import ExecutionStats  # for typing
        except ImportError:
            raise RuntimeError("This class requires at least pytorch version 1.2.0")

        # My typing extension PyRight doesn't support this complex
        # comment type annotation
        self._bench = None  # type: Optional[ThroughputBenchmark]
        self.execution_stats = None  # type: Optional[ExecutionStats]

    def run(
        self,
        *args,
        max_batches: int = 10,
        num_calling_threads: int = 1,
        num_warmup_iters: int = 10,
        num_iters: int = 100,
        **kwargs
    ):
        @self.on(Events.ITERATION_COMPLETED(once=max_batches))
        def terminate_after_max_batches(engine):
            engine.terminate()

        super().run(*args, **kwargs)
        self.execution_stats = self._bench.benchmark(
            num_calling_threads=num_calling_threads, num_warmup_iters=num_warmup_iters, num_iters=num_iters
        )
        return self.execution_stats


def create_throughput_benchmark(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_input: Callable = _prepare_input,
) -> BenchmarkEngine:
    try:
        from torch.utils.throughput_benchmark import ThroughputBenchmark
    except ImportError:
        raise RuntimeError("This method requires at least pytorch version 1.2.0")

    def _run(engine: BenchmarkEngine, batch: Sequence[torch.Tensor]) -> None:
        # In this loop nothing happens besides the loading of the input data
        model.eval()  # Should the user expect the automatic setting to eval?
        input_data = _prepare_input(batch, device=device, non_blocking=non_blocking)
        engine._bench.add_input(input_data)

    engine = BenchmarkEngine(_run)

    # see PR https://github.com/pytorch/ignite/pull/835
    if device is not None:

        @engine.on(Events.STARTED)
        def move_device(engine):
            model.to(device)

    # Load initialize benchmark before loading model
    @engine.on(Events.STARTED)
    def create_benchmark(benchmark_engine: BenchmarkEngine):
        engine._bench = ThroughputBenchmark(model)

    return engine
