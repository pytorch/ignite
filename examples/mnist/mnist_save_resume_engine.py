from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from tqdm import tqdm
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, DiskSaver


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
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def run(train_batch_size, val_batch_size,
        epochs, lr, momentum,
        log_interval, log_dir,
        checkpoint_every,
        resume_from, crash_iteration=1000):

    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = Net()
    writer = SummaryWriter(logdir=log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    criterion = nn.NLLLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(criterion)},
                                            device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_step(engine):
        lr_scheduler.step()

    desc = "ITERATION - loss: {:.4f} - lr: {:.4f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0, lr)
    )

    if log_interval is None:
        e = Events.ITERATION_COMPLETED
        log_interval = 1
    else:
        e = Events.ITERATION_COMPLETED(every=log_interval)

    @trainer.on(e)
    def log_training_loss(engine):
        lr = optimizer.param_groups[0]['lr']
        pbar.desc = desc.format(engine.state.output, lr)
        pbar.update(log_interval)
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        writer.add_scalar("lr", lr, engine.state.iteration)

    if resume_from is None:
        @trainer.on(Events.ITERATION_COMPLETED(once=crash_iteration))
        def _(engine):
            raise Exception("STOP at {}".format(engine.state.iteration))
    else:
        @trainer.on(Events.STARTED)
        def _(engine):
            pbar.n = engine.state.iteration

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))
        pbar.n = pbar.last_print_n = 0
        writer.add_scalar("valdation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)

    objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    training_checkpoint = Checkpoint(to_save=objects_to_checkpoint,
                                     save_handler=DiskSaver(log_dir, require_empty=False))

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_every), training_checkpoint)

    if resume_from is not None:
        tqdm.write("Resume from a checkpoint: {}".format(resume_from))
        checkpoint = torch.load(resume_from)
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)

    try:
        trainer.run(train_loader, max_epochs=epochs)
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    pbar.close()
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="/tmp/mnist_save_resume",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--checkpoint_every', type=int, default=550, help='Checkpoint training every X iterations')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to the checkpoint .pth file to resume training from')
    parser.add_argument('--crash_iteration', type=int, default=3000, help='Iteration at which to raise an exception')

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size,
        args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir, args.checkpoint_every,
        args.resume_from, args.crash_iteration)
