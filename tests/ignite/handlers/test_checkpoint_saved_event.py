import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers.checkpoint import Checkpoint

def test_saved_checkpoint_event():
    """Test that saved_checkpoint event fires when checkpoints are saved"""
    # Create a simple model and data
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Dummy dataset
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Create trainer
    trainer = Engine(train_step)

    # Register the event
    trainer.register_events("saved_checkpoint")

    checkpoint_saves = []

    @trainer.on("saved_checkpoint")
    def on_checkpoint_saved(engine):
        checkpoint_saves.append(f"Epoch {engine.state.epoch}, Iteration {engine.state.iteration}")

    # Create checkpoint handler
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_handler = Checkpoint(
            {'model': model, 'optimizer': optimizer},
            tmpdir,
            filename_prefix='training',
            n_saved=2
        )
        
        # Save every 2 epochs
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), checkpoint_handler)
        
        trainer.run(dataloader, max_epochs=5)
        
        # ASSERTIONS (this is the key change!)
        assert len(checkpoint_saves) == 2, f"Expected 2 checkpoint saves, got {len(checkpoint_saves)}"
        assert "Epoch 2" in checkpoint_saves[0], "First checkpoint should be at epoch 2"
        assert "Epoch 4" in checkpoint_saves[1], "Second checkpoint should be at epoch 4"