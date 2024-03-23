import argparse
import os
import random
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from ignite.engine import Engine, Events

from ignite.handlers import ModelCheckpoint, ProgressBar, Timer
from ignite.metrics import RunningAverage

try:
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torchvision.utils as vutils

except ImportError:
    raise ModuleNotFoundError(
        "Please install torchvision to run this example, for example "
        "via conda by running 'conda install -c pytorch torchvision'. "
    )


PRINT_FREQ = 100
FAKE_IMG_FNAME = "fake_sample_epoch_{:04d}.png"
REAL_IMG_FNAME = "real_sample_epoch_{:04d}.png"
LOGS_FNAME = "logs.tsv"
PLOT_FNAME = "plot.svg"
SAMPLES_FNAME = "samples.svg"
CKPT_PREFIX = "networks"


class Net(nn.Module):
    """A base class for both generator and the discriminator.
    Provides a common weight initialization scheme.

    """

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if "Conv" in classname:
                m.weight.data.normal_(0.0, 0.02)

            elif "BatchNorm" in classname:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return x


class Generator(Net):
    """Generator network.

    Args:
        nf (int): Number of filters in the second-to-last deconv layer
    """

    def __init__(self, z_dim, nf, nc):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=nf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),
            # state size. (nf*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),
            # state size. (nf*4) x 8 x 8
            nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),
            # state size. (nf*2) x 16 x 16
            nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            # state size. (nf) x 32 x 32
            nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

        self.weights_init()

    def forward(self, x):
        return self.net(x)


class Discriminator(Net):
    """Discriminator network.

    Args:
        nf (int): Number of filters in the first conv layer.
    """

    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf) x 32 x 32
            nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*2) x 16 x 16
            nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*4) x 8 x 8
            nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*8) x 4 x 4
            nn.Conv2d(in_channels=nf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.weights_init()

    def forward(self, x):
        output = self.net(x)
        return output.view(-1, 1).squeeze(1)


def check_manual_seed(seed):
    """If manual seed is not specified, choose a random one and communicate it to the user."""

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Using manual seed: {seed}")


def check_dataset(dataset, dataroot):
    """

    Args:
        dataset (str): Name of the dataset to use. See CLI help for details
        dataroot (str): root directory where the dataset will be stored.

    Returns:
        dataset (data.Dataset): torchvision Dataset object

    """
    resize = transforms.Resize(64)
    crop = transforms.CenterCrop(64)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    if dataset in {"imagenet", "folder", "lfw"}:
        dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([resize, crop, to_tensor, normalize]))
        nc = 3

    elif dataset == "lsun":
        dataset = dset.LSUN(
            root=dataroot, classes=["bedroom_train"], transform=transforms.Compose([resize, crop, to_tensor, normalize])
        )
        nc = 3

    elif dataset == "cifar10":
        dataset = dset.CIFAR10(
            root=dataroot, download=True, transform=transforms.Compose([resize, to_tensor, normalize])
        )
        nc = 3

    elif dataset == "mnist":
        dataset = dset.MNIST(root=dataroot, download=True, transform=transforms.Compose([resize, to_tensor, normalize]))
        nc = 1

    elif dataset == "fake":
        dataset = dset.FakeData(size=256, image_size=(3, 64, 64), transform=to_tensor)
        nc = 3

    else:
        raise RuntimeError(f"Invalid dataset name: {dataset}")

    return dataset, nc


def main(
    dataset,
    dataroot,
    z_dim,
    g_filters,
    d_filters,
    batch_size,
    epochs,
    learning_rate,
    beta_1,
    saved_G,
    saved_D,
    seed,
    n_workers,
    device,
    alpha,
    output_dir,
):
    # seed
    check_manual_seed(seed)

    # data
    dataset, num_channels = check_dataset(dataset, dataroot)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)

    # netowrks
    netG = Generator(z_dim, g_filters, num_channels).to(device)
    netD = Discriminator(num_channels, d_filters).to(device)

    # criterion
    bce = nn.BCELoss()

    # optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

    # load pre-trained models
    if saved_G:
        netG.load_state_dict(torch.load(saved_G))

    if saved_D:
        netD.load_state_dict(torch.load(saved_D))

    # misc
    real_labels = torch.ones(batch_size, device=device)
    fake_labels = torch.zeros(batch_size, device=device)
    fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)

    def get_noise():
        return torch.randn(batch_size, z_dim, 1, 1, device=device)

    # The main function, processing a batch of examples
    def step(engine, batch):
        # unpack the batch. It comes from a dataset, so we have <images, labels> pairs. Discard labels.
        real, _ = batch
        real = real.to(device)

        # -----------------------------------------------------------
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()

        # train with real
        output = netD(real)
        errD_real = bce(output, real_labels)
        D_x = output.mean().item()

        errD_real.backward()

        # get fake image from generator
        noise = get_noise()
        fake = netG(noise)

        # train with fake
        output = netD(fake.detach())
        errD_fake = bce(output, fake_labels)
        D_G_z1 = output.mean().item()

        errD_fake.backward()

        # gradient update
        errD = errD_real + errD_fake
        optimizerD.step()

        # -----------------------------------------------------------
        # (2) Update G network: maximize log(D(G(z)))
        netG.zero_grad()

        # Update generator. We want to make a step that will make it more likely that discriminator outputs "real"
        output = netD(fake)
        errG = bce(output, real_labels)
        D_G_z2 = output.mean().item()

        errG.backward()

        # gradient update
        optimizerG.step()

        return {"errD": errD.item(), "errG": errG.item(), "D_x": D_x, "D_G_z1": D_G_z1, "D_G_z2": D_G_z2}

    # ignite objects
    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(output_dir, CKPT_PREFIX, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    # attach running average metrics
    monitoring_metrics = ["errD", "errG", "D_x", "D_G_z1", "D_G_z2"]
    RunningAverage(alpha=alpha, output_transform=lambda x: x["errD"]).attach(trainer, "errD")
    RunningAverage(alpha=alpha, output_transform=lambda x: x["errG"]).attach(trainer, "errG")
    RunningAverage(alpha=alpha, output_transform=lambda x: x["D_x"]).attach(trainer, "D_x")
    RunningAverage(alpha=alpha, output_transform=lambda x: x["D_G_z1"]).attach(trainer, "D_G_z1")
    RunningAverage(alpha=alpha, output_transform=lambda x: x["D_G_z2"]).attach(trainer, "D_G_z2")

    # attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    @trainer.on(Events.ITERATION_COMPLETED(every=PRINT_FREQ))
    def print_logs(engine):
        fname = output_dir / LOGS_FNAME
        columns = ["iteration"] + list(engine.state.metrics.keys())
        values = [str(engine.state.iteration)] + [str(round(value, 5)) for value in engine.state.metrics.values()]

        with open(fname, "a") as f:
            if f.tell() == 0:
                print("\t".join(columns), file=f)
            print("\t".join(values), file=f)
        message = f"[{engine.state.epoch}/{epochs}][{engine.state.iteration % len(loader)}/{len(loader)}]"
        for name, value in zip(columns, values):
            message += f" | {name}: {value}"

        pbar.log_message(message)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_fake_example(engine):
        fake = netG(fixed_noise)
        path = output_dir / FAKE_IMG_FNAME.format(engine.state.epoch)
        vutils.save_image(fake.detach(), path, normalize=True)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_real_example(engine):
        img, y = engine.state.batch
        path = output_dir / REAL_IMG_FNAME.format(engine.state.epoch)
        vutils.save_image(img, path, normalize=True)

    # adding handlers using `trainer.add_event_handler` method API
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={"netG": netG, "netD": netD}
    )

    # automatically adding handlers via a special `attach` method of `Timer` handler
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
        timer.reset()

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def create_plots(engine):
        try:
            import matplotlib as mpl

            mpl.use("agg")

            import matplotlib.pyplot as plt
            import pandas as pd

        except ImportError:
            warnings.warn("Loss plots will not be generated -- pandas or matplotlib not found")

        else:
            df = pd.read_csv(output_dir / LOGS_FNAME, delimiter="\t", index_col="iteration")
            _ = df.plot(subplots=True, figsize=(20, 20))
            _ = plt.xlabel("Iteration number")
            fig = plt.gcf()
            path = output_dir / PLOT_FNAME

            fig.savefig(path)

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn("KeyboardInterrupt caught. Exiting gracefully.")

            create_plots(engine)
            checkpoint_handler(engine, {"netG_exception": netG, "netD_exception": netD})

        else:
            raise e

    # Setup is done. Now let's run the training
    trainer.run(loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices={"cifar10", "lsun", "imagenet", "folder", "lfw", "fake", "mnist"},
        help="Type of the dataset to be used.",
    )

    parser.add_argument("--dataroot", required=True, help="path to dataset")

    parser.add_argument("--workers", type=int, default=2, help="number of data loading workers")

    parser.add_argument("--batch-size", type=int, default=64, help="input batch size")

    parser.add_argument("--z-dim", type=int, default=100, help="size of the latent z vector")

    parser.add_argument(
        "--g-filters", type=int, default=64, help="Number of filters in the second-to-last generator deconv layer"
    )

    parser.add_argument("--d-filters", type=int, default=64, help="Number of filters in first discriminator conv layer")

    parser.add_argument("--epochs", type=int, default=25, help="number of epochs to train for")

    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")

    parser.add_argument("--beta-1", type=float, default=0.5, help="beta_1 for adam")

    parser.add_argument("--no-cuda", action="store_true", help="disables cuda")

    parser.add_argument("--saved-G", default="", help="path to pickled generator (to continue training)")

    parser.add_argument("--saved-D", default="", help="path to pickled discriminator (to continue training)")

    parser.add_argument("--output-dir", default=".", help="directory to output images and model checkpoints")

    parser.add_argument("--seed", type=int, help="manual seed")

    parser.add_argument("--alpha", type=float, default=0.98, help="smoothing constant for exponential moving averages")

    args = parser.parse_args()
    dev = "cpu" if (not torch.cuda.is_available() or args.no_cuda) else "cuda:0"

    args.output_dir = Path(args.output_dir)
    try:
        args.output_dir.mkdir(parents=True)
    except FileExistsError:
        if (not args.output_dir.is_dir()) or (len(os.listdir(args.output_dir)) > 0):
            raise FileExistsError("Please provide a path to a non-existing or empty directory.")

    main(
        dataset=args.dataset,
        dataroot=args.dataroot,
        z_dim=args.z_dim,
        g_filters=args.g_filters,
        d_filters=args.d_filters,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        beta_1=args.beta_1,
        saved_D=args.saved_D,
        saved_G=args.saved_G,
        seed=args.seed,
        device=dev,
        n_workers=args.workers,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
