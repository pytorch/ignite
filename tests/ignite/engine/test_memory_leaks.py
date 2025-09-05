import sys
import weakref
import torch
import torch.nn as nn
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import ProgressBar, TensorboardLogger
from ignite.handlers.tensorboard_logger import OptimizerParamsHandler
from torch.optim import Adam
from ignite.metrics import Loss
from torch.utils.data import DataLoader, TensorDataset


class TestEngineMemoryLeak:
    ENGINE_WEAK_REFS = {}

    def do(self, model, dataloader, device, runs_folder):
        optim = Adam(model.parameters(), 1e-4)
        loss = nn.BCEWithLogitsLoss()
        trainer = create_supervised_trainer(model, optim, loss, device)
        metrics = {"Loss": Loss(loss)}
        evaluator = create_supervised_evaluator(model, metrics, device)

        pbar = ProgressBar()
        pbar.attach(trainer)

        tb_logger = TensorboardLogger(log_dir=runs_folder)
        tb_logger.attach(trainer, OptimizerParamsHandler(optim), Events.EPOCH_STARTED)

        trainer.run(dataloader, 1)

        @trainer.on(Events.COMPLETED)
        def completed(engine):
            evaluator.run(dataloader)

        tb_logger.close()
        pbar.close()

        self.ENGINE_WEAK_REFS[weakref.ref(trainer)] = sys.getrefcount(trainer) - 1
        self.ENGINE_WEAK_REFS[weakref.ref(evaluator)] = sys.getrefcount(evaluator) - 1

    def test_circular_references(self, tmp_path):
        runs_folder = tmp_path / "runs"
        runs_folder.mkdir()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        x = torch.rand(32, 1, 64, 64, 32)
        y = torch.round(torch.rand(32, 1))
        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, 6)
        for i in range(5):
            N = 3000
            model = nn.Sequential(nn.Flatten(), nn.Linear(64 * 64 * 32, N), nn.ReLU(), nn.Linear(N, 1))
            model = model.to(device)
            self.do(model, dataloader, device, runs_folder)
            for engine_weak_ref, val in self.ENGINE_WEAK_REFS.items():
                engine = engine_weak_ref()
                if engine is not None:
                    ref_count = sys.getrefcount(engine) - 1
                    error_message = f"Engine Memory Leak: {engine} - Ref Count: {ref_count}"
                    print(error_message)
                    assert ref_count == 0

            print("!!!", i, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
