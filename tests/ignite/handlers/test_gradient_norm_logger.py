import torch
from ignite.engine import Engine
from ignite.handlers import GradientNormLogger


def _dummy_step(engine, batch):
    model = engine.state.model
    optimizer = engine.state.optimizer

    optimizer.zero_grad()
    output = model(batch)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    return loss.item()


def test_gradient_norm_logger():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    engine = Engine(_dummy_step)
    engine.state.model = model
    engine.state.optimizer = optimizer

    grad_logger = GradientNormLogger(model)
    grad_logger.attach(engine)

    data = [torch.randn(10) for _ in range(5)]

    engine.run(data, max_epochs=1)

    assert "grad_norm" in engine.state.metrics
    assert engine.state.metrics["grad_norm"] >= 0
