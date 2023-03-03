from __future__ import print_function

import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_test_set, get_training_set
from model import Net
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint
from ignite.metrics import PSNR

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Super Res Example")
parser.add_argument("--upscale_factor", type=int, required=True, help="super resolution upscale factor")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=10, help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=2, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate. Default=0.01")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--mps", action="store_true", default=False, help="enables macOS GPU training")
parser.add_argument("--threads", type=int, default=4, help="number of threads for data loader to use")
parser.add_argument("--seed", type=int, default=123, help="random seed to use. Default=123")
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
if not opt.mps and torch.backends.mps.is_available():
    raise Exception("Found mps device, please run with --mps to enable macOS GPU")

torch.manual_seed(opt.seed)
use_mps = opt.mps and torch.backends.mps.is_available()

if opt.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("===> Loading datasets")
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print("===> Building model")
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train_step(engine, batch):
    input, target = batch[0].to(device), batch[1].to(device)

    optimizer.zero_grad()
    loss = criterion(model(input), target)
    loss.backward()
    optimizer.step()

    return loss.item()


def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)

    return y_pred, y


trainer = Engine(train_step)
evaluator = Engine(validation_step)
psnr = PSNR(data_range=1)
psnr.attach(evaluator, "psnr")
validate_every = 1
log_interval = 10


@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(
        "===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
            engine.state.epoch, engine.state.iteration, len(training_data_loader), engine.state.output
        )
    )


@trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
def run_validation():
    evaluator.run(testing_data_loader)


@trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
def log_validation():
    metrics = evaluator.state.metrics
    print(f"Epoch: {trainer.state.epoch}, Avg. PSNR: {metrics['psnr']} dB")


@trainer.on(Events.EPOCH_COMPLETED)
def checkpoint():
    model_out_path = "model_epoch_{}.pth".format(trainer.state.epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


trainer.run(training_data_loader, opt.nEpochs)
