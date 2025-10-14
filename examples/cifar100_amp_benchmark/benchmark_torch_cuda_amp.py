import fire
import torch
from torch.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import wide_resnet50_2
from utils import get_train_eval_loaders

from ignite.engine import convert_tensor, create_supervised_evaluator, Engine, Events

from ignite.handlers import ProgressBar, Timer
from ignite.metrics import Accuracy, Loss


def main(dataset_path, batch_size=256, max_epochs=10):
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
    torch.backends.cudnn.benchmark = True

    device = "cuda"

    train_loader, test_loader, eval_train_loader = get_train_eval_loaders(dataset_path, batch_size=batch_size)

    model = wide_resnet50_2(num_classes=100).to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss().to(device)

    scaler = GradScaler()

    def train_step(engine, batch):
        x = convert_tensor(batch[0], device, non_blocking=True)
        y = convert_tensor(batch[1], device, non_blocking=True)

        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast("cuda"):
            y_pred = model(x)
            loss = criterion(y_pred, y)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same precision that autocast used for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        return loss.item()

    trainer = Engine(train_step)
    timer = Timer(average=True)
    timer.attach(trainer, step=Events.EPOCH_COMPLETED)
    ProgressBar(persist=True).attach(trainer, output_transform=lambda out: {"batch loss": out})

    metrics = {"Accuracy": Accuracy(), "Loss": Loss(criterion)}

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def log_metrics(engine, title):
        for name in metrics:
            print(f"\t{title} {name}: {engine.state.metrics[name]:.2f}")

    @trainer.on(Events.COMPLETED)
    def run_validation(_):
        print(f"- Mean elapsed time for 1 epoch: {timer.value()}")
        print("- Metrics:")
        with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "Train"):
            evaluator.run(eval_train_loader)

        with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "Test"):
            evaluator.run(test_loader)

    trainer.run(train_loader, max_epochs=max_epochs)


if __name__ == "__main__":
    fire.Fire(main)
