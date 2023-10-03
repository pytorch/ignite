from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import manual_seed

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ModuleNotFoundError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice (pip or conda)."
        )


# Basic model's definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    """Method to setup data loaders: train_loader and val_loader"""
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, val_loader


def log_model_weights(engine, model=None, fp=None, **kwargs):
    """Helper method to log norms of model weights: print and dump into a file"""
    assert model and fp
    output = {"total": 0.0}
    max_counter = 5
    for name, p in model.named_parameters():
        name = name.replace(".", "/")
        n = torch.norm(p)
        if max_counter > 0:
            output[name] = n
        output["total"] += n
        max_counter -= 1
    output_items = " - ".join([f"{m}:{v:.4f}" for m, v in output.items()])
    msg = f"{engine.state.epoch} | {engine.state.iteration}: {output_items}"

    with open(fp, "a") as h:
        h.write(msg)
        h.write("\n")


def log_model_grads(engine, model=None, fp=None, **kwargs):
    """Helper method to log norms of model gradients: print and dump into a file"""
    assert model and fp
    output = {"grads/total": 0.0}
    max_counter = 5
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        name = name.replace(".", "/")
        n = torch.norm(p.grad)
        if max_counter > 0:
            output[f"grads/{name}"] = n
        output["grads/total"] += n
        max_counter -= 1

    output_items = " - ".join([f"{m}:{v:.4f}" for m, v in output.items()])
    msg = f"{engine.state.epoch} | {engine.state.iteration}: {output_items}"

    with open(fp, "a") as h:
        h.write(msg)
        h.write("\n")


def log_data_stats(engine, fp=None, **kwargs):
    """Helper method to log mean/std of input batch of images and median of batch of targets."""
    assert fp
    x, y = engine.state.batch
    output = {
        "batch xmean": x.mean().item(),
        "batch xstd": x.std().item(),
        "batch ymedian": y.median().item(),
    }
    output_items = " - ".join([f"{m}:{v:.4f}" for m, v in output.items()])
    msg = f"{engine.state.epoch} | {engine.state.iteration}: {output_items}"

    with open(fp, "a") as h:
        h.write(msg)
        h.write("\n")


def run(
    train_batch_size,
    val_batch_size,
    epochs,
    lr,
    momentum,
    log_interval,
    log_dir,
    checkpoint_every,
    resume_from,
    crash_iteration=-1,
    deterministic=False,
):
    # Setup seed to have same model's initialization:
    manual_seed(75)

    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    writer = SummaryWriter(log_dir=log_dir)
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    model.to(device)  # Move model before creating optimizer
    criterion = nn.NLLLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    # Setup trainer and evaluator
    if deterministic:
        tqdm.write("Setup deterministic trainer")
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, deterministic=deterministic)
    running_loss = RunningAverage(output_transform=lambda x: x)
    running_loss.attach(trainer, "rloss")

    metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, metrics, device)

    # Apply learning rate scheduling
    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_step(engine):
        lr_scheduler.step()

    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=f"Epoch {0} - loss: {0:.4f} - lr: {lr:.4f}")

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        lr = optimizer.param_groups[0]["lr"]
        rloss = engine.state.metrics["rloss"]
        pbar.desc = f"Epoch {engine.state.epoch} - loss: {rloss:.4f} - lr: {lr:.4f}"
        pbar.update(log_interval)
        writer.add_scalar("training/running_loss", rloss, engine.state.iteration)
        writer.add_scalar("lr", lr, engine.state.iteration)

    if crash_iteration > 0:

        @trainer.on(Events.ITERATION_COMPLETED(once=crash_iteration))
        def _(engine):
            raise Exception(f"STOP at {engine.state.iteration}")

    if resume_from is not None:

        @trainer.on(Events.STARTED)
        def _(engine):
            pbar.n = engine.state.iteration % engine.state.epoch_length

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    # Compute and log validation metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )
        pbar.n = pbar.last_print_n = 0
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # Setup object to checkpoint
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "train_running_loss": running_loss,
        "metrics": metrics,
    }
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(log_dir, require_empty=False),
        n_saved=None,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_every), training_checkpoint)

    # Setup logger to print and dump into file: model weights, model grads and data stats
    # - first 3 iterations
    # - 4 iterations after checkpointing
    # This helps to compare resumed training with checkpointed training
    def log_event_filter(e, event):
        if event in [1, 2, 3]:
            return True
        elif 0 <= (event % (checkpoint_every * e.state.epoch_length)) < 5:
            return True
        return False

    fp = Path(log_dir) / ("run.log" if resume_from is None else "resume_run.log")
    fp = fp.as_posix()
    for h in [log_data_stats, log_model_weights, log_model_grads]:
        trainer.add_event_handler(Events.ITERATION_COMPLETED(event_filter=log_event_filter), h, model=model, fp=fp)

    if resume_from is not None:
        tqdm.write(f"Resume from the checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

    try:
        # Synchronize random states
        manual_seed(15)
        trainer.run(train_loader, max_epochs=epochs)
    except Exception as e:
        import traceback

        print(traceback.format_exc())

    pbar.close()
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument(
        "--val_batch_size", type=int, default=1000, help="input batch size for validation (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--log_dir", type=str, default="/tmp/mnist_save_resume", help="log directory for Tensorboard log output"
    )
    parser.add_argument("--checkpoint_every", type=int, default=1, help="Checkpoint training every X epochs")
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Path to the checkpoint .pt file to resume training from"
    )
    parser.add_argument("--crash_iteration", type=int, default=-1, help="Iteration at which to raise an exception")
    parser.add_argument(
        "--deterministic", action="store_true", help="Deterministic training with dataflow synchronization"
    )

    args = parser.parse_args()

    run(
        args.batch_size,
        args.val_batch_size,
        args.epochs,
        args.lr,
        args.momentum,
        args.log_interval,
        args.log_dir,
        args.checkpoint_every,
        args.resume_from,
        args.crash_iteration,
        args.deterministic,
    )
