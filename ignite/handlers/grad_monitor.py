import math
from collections import deque
from typing import Callable, List, Optional

import torch
from torch.nn import Module

from ignite import distributed as idist
from ignite.engine import Engine, Events, engine
from ignite.utils import setup_logger

__all__ = ["GradMonitor"]

class GradMonitor:
    """
    Monitors the L2 gradient norm to catch training instability early.
    
    If the norm exceeds the threshold, we log a warning and run an optional callback.
    This is especially useful for transformers where gradients can explode suddenly.

    .. versionadded:: 0.5.3

    Example:

    .. code-block:: python

        from ignite.handlers import GradMonitor

        # Initialize the monitor
        monitor = GradMonitor(model, threshold=10.0)
        monitor.attach(trainer)

        # Implementation for batch skipping
        @trainer.on(Events.ITERATION_COMPLETED)
        def skip_batch_on_spike(engine):
            if getattr(engine.state, "unhealthy_spike", False):
                # Custom logic here, e.g., zeroing grads or logging
                optimizer.zero_grad()
                print(f"Spike detected at iteration {engine.state.iteration}, skipping.")

    Args:
        model: The torch.nn.Module whose gradients are to be monitored.
        threshold: The L2 norm value above which a spike is flagged. Default: 100.0.
        window_size: Number of iterations for moving average in dynamic mode.
        use_dynamic: If True, uses a dynamic threshold (5x moving average).
        callback: Optional function called when a spike is detected.
    """

    def __init__(
        self,
        model: Module,
        threshold: float = 100.0,
        window_size: int = 10,
        use_dynamic: bool = False,
        callback: Optional[Callable[[Engine, float], None]] = None,
    ) -> None:
        if not isinstance(model, Module):
            raise TypeError(f"Argument model should be a torch.nn.Module, but given {type(model)}.")

        if not (isinstance(threshold, (int, float)) and math.isfinite(threshold) and threshold > 0):
            raise ValueError(f"Argument threshold should be a positive finite number, but given {threshold}.")

        if callback is not None and not callable(callback):
            raise TypeError(f"Argument callback should be callable, but given {type(callback)}.")

        self.model = model
        self.threshold = float(threshold)
        self.use_dynamic = use_dynamic
        self.callback = callback
        
        # Standard Ignite logger setup.
        self.logger = setup_logger(__name__ + "." + self.__class__.__name__)
        self._history = deque(maxlen=1000)
        self._window = deque(maxlen=window_size) 

    def attach(self, engine: Engine) -> None:
        # Attaches the monitor to the trainer's iteration cycle.
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def _compute_grad_norm(self, engine: Optional[Engine] = None) -> float:
        # Calculates the total L2 norm,Handling DDP synchronization and AMP scaling.
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                # Uses .data.norm to get raw scalar value quickly.
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2

        # DDP Sync: Sums squared norms across all GPUs so all nodes see the same 'spike'.
        if idist.get_world_size() > 1:
            total_norm_sq = idist.all_reduce(total_norm_sq)

        total_norm = total_norm_sq ** 0.5

        # Handles GradScaler for Automatic Mixed Precision (AMP) workflows.
        if engine is not None and hasattr(engine, "scaler"):
            scaler = getattr(engine, "scaler")
            if scaler is not None:
                total_norm /= scaler.get_scale()

        return total_norm   

    def __call__(self, engine: Engine) -> bool:
        # Main execution logic triggered at the end of each iteration.
        grad_norm = self._compute_grad_norm(engine)
        self._history.append(grad_norm)
        self._window.append(grad_norm)        

        # Logic for dynamic thresholding.
        effective_threshold = self.threshold
        if self.use_dynamic and len(self._window) == self._window.maxlen:
            # Calculates spike relative to the moving average of the last 'n' steps.
            avg_norm = sum(self._window) / len(self._window)
            effective_threshold = avg_norm * 5.0  # Spike is defined as 5x the average.
        if grad_norm > effective_threshold:
            self.logger.warning(
                "Iteration %d: Gradient Spike Detected (%.2f > %.2f)", 
                engine.state.iteration, grad_norm, effective_threshold
            )
            engine.state.unhealthy_spike = True
            
            if self.callback:
                self.callback(engine, grad_norm)
            return True 

        engine.state.unhealthy_spike = False
        return False

    @property
    def history(self) -> List[float]:
        """Provides access to the historical gradient norms for analysis or plotting."""
        return list(self._history)