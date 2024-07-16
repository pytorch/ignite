import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import Net
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop, resize, to_tensor

from ignite.engine import Engine, Events

from ignite.handlers import BasicTimeProfiler, ProgressBar
from ignite.metrics import PSNR

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Super Res Example")
parser.add_argument("--crop_size", type=int, default=256, help="cropped size of the images for training")
parser.add_argument("--upscale_factor", type=int, required=True, help="super resolution upscale factor")
parser.add_argument("--batch_size", type=int, default=64, help="training batch size")
parser.add_argument("--test_batch_size", type=int, default=10, help="testing batch size")
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate. Default=0.01")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--mps", action="store_true", default=False, help="enables macOS GPU training")
parser.add_argument("--threads", type=int, default=4, help="number of threads for data loader to use")
parser.add_argument("--seed", type=int, default=123, help="random seed to use. Default=123")
parser.add_argument("--debug", action="store_true", help="use debug")

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


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scale_factor, crop_size=256):
        self.dataset = dataset
        self.scale_factor = scale_factor
        self.crop_size = crop_size

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        img = image.convert("YCbCr")
        hr_image, _, _ = img.split()
        hr_image = center_crop(hr_image, self.crop_size)
        lr_image = hr_image.copy()
        if self.scale_factor != 1:
            size = self.crop_size // self.scale_factor
            lr_image = resize(lr_image, [size, size])
        hr_image = to_tensor(hr_image)
        lr_image = to_tensor(lr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.dataset)


try:
    trainset = torchvision.datasets.Caltech101(root="./data", download=True)
    testset = torchvision.datasets.Caltech101(root="./data", download=False)
except RuntimeError:
    print("Dataset download problem, exiting without error code")
    exit(0)

trainset_sr = SRDataset(trainset, scale_factor=opt.upscale_factor, crop_size=opt.crop_size)
testset_sr = SRDataset(testset, scale_factor=opt.upscale_factor, crop_size=opt.crop_size)

training_data_loader = DataLoader(dataset=trainset_sr, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=testset_sr, num_workers=opt.threads, batch_size=opt.test_batch_size)

print("===> Building model")
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train_step(engine, batch):
    model.train()
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

if opt.debug:
    epoch_length = 10
    validate_epoch_length = 1
else:
    epoch_length = len(training_data_loader)
    validate_epoch_length = len(testing_data_loader)


@trainer.on(Events.EPOCH_COMPLETED(every=validate_every))
def log_validation():
    evaluator.run(testing_data_loader, epoch_length=validate_epoch_length)
    metrics = evaluator.state.metrics
    print(f"Epoch: {trainer.state.epoch}, Avg. PSNR: {metrics['psnr']} dB")


@trainer.on(Events.EPOCH_COMPLETED)
def checkpoint():
    model_out_path = "model_epoch_{}.pth".format(trainer.state.epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# Attach basic profiler
basic_profiler = BasicTimeProfiler()
basic_profiler.attach(trainer)

ProgressBar().attach(trainer, output_transform=lambda x: {"loss": x})

trainer.run(training_data_loader, opt.n_epochs, epoch_length=epoch_length)

results = basic_profiler.get_results()
basic_profiler.print_results(results)
