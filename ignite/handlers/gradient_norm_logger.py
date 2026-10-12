from typing import Any, Optional

import torch
from ignite.engine import Engine, Events


def compute_grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """Compute gradient norm of a model."""
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    return total_norm ** (1.0 / norm_type)


class GradientNormLogger:
    """Handler to log gradient norm during training.

    Args:
        model: PyTorch model
        norm_type: Type of norm (default: L2)
        output_transform: Optional transform applied to output
    """

    def __init__(
        self,
        model: torch.nn.Module,
        norm_type: float = 2.0,
        output_transform: Optional[Any] = None,
    ):
        self.model = model
        self.norm_type = norm_type
        self.output_transform = output_transform

    def __call__(self, engine: Engine) -> None:
        norm = compute_grad_norm(self.model, self.norm_type)

        if not hasattr(engine.state, "metrics"):
            engine.state.metrics = {}

        engine.state.metrics["grad_norm"] = norm

    def attach(self, engine: Engine, event_name: Events = Events.ITERATION_COMPLETED) -> None:
        engine.add_event_handler(event_name, self)
