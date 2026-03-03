import pytest
import time
import math
import torch
from ignite.engine import Engine
from ignite.handlers.grad_monitor import GradMonitor

def test_gradient_norm_math():
    # Verifies that L2 norm calculation matches theoretical values.
    layer = torch.nn.Linear(10, 5)
    monitor = GradMonitor(layer)
    for p in layer.parameters():
        p.grad = torch.ones_like(p)
    
    expected_params = sum(p.numel() for p in layer.parameters())
    calculated_norm = monitor._compute_grad_norm()
    assert torch.isclose(torch.tensor(calculated_norm), torch.tensor(expected_params**0.5))

def test_engine_state_flag():
    # Verifies that unhealthy_spike flag is correctly set/reset on engine state.
    model = torch.nn.Linear(5, 1)
    monitor = GradMonitor(model, threshold=0.1)
    trainer = Engine(lambda e, b: None)
    
    # Simulating a spike.
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 10.0
    monitor(trainer)
    assert trainer.state.unhealthy_spike is True

    # Simulating healthy gradients.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    monitor(trainer)
    assert trainer.state.unhealthy_spike is False

def test_grad_scaler_unscaling():
    # Verifying gradient unscaling when a GradScaler (AMP) is present.
    model = torch.nn.Linear(10, 1)
    monitor = GradMonitor(model)
    trainer = Engine(lambda e, b: None)
    
    class MockScaler:
        def get_scale(self): return 1024.0
    
    trainer.scaler = MockScaler()
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 1024.0 
        
    calculated_norm = monitor._compute_grad_norm(trainer)
    expected_norm = sum(p.numel() for p in model.parameters())**0.5
    assert math.isclose(calculated_norm, expected_norm, rel_tol=1e-5)

def test_input_validation():
    # Verifying initialization checks for invalid parameters.
    with pytest.raises(TypeError, match="should be a torch.nn.Module"):
        GradMonitor(model=None)
    
    dummy_model = torch.nn.Linear(1, 1)
    for bad_threshold in [-5.0, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match="positive finite number"):
            GradMonitor(dummy_model, threshold=bad_threshold)

def benchmark_overhead():
    model = torch.nn.Sequential(torch.nn.Linear(1000, 1000)) 
    monitor = GradMonitor(model)
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    iterations = 1000
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = monitor._compute_grad_norm()
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    total_params = sum(p.numel() for p in model.parameters())
    ns_per_param = ((end_time - start_time) / (iterations * total_params)) * 1e9
    
    print(f"\n--- BENCHMARK ---")
    print(f"Average time per iteration: {avg_time_ms:.6f} ms")
    print(f"Approximate overhead per parameter: {ns_per_param:.2f} ns")

if __name__ == "__main__":
    print("Running Logic Tests...")
    pytest.main([__file__, "-v"])
    benchmark_overhead()