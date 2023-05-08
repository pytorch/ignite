# coding: utf-8
import argparse
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import utils
from handlers import Progbar
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformer_net import TransformerNet
from vgg import Vgg16

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint


def check_paths(args):
    try:
        if args.checkpoint_model_dir is not None and not (Path(args.checkpoint_model_dir).exists()):
            Path(args.checkpoint_model_dir).mkdir(parents=True)
    except OSError as e:
        raise OSError(e)


def check_manual_seed(args):
    seed = args.seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_dataset(args):
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ]
    )

    if args.dataset in {"folder", "mscoco"}:
        train_dataset = datasets.ImageFolder(args.dataroot, transform)
    elif args.dataset == "test":
        train_dataset = datasets.FakeData(
            size=args.batch_size, image_size=(3, 32, 32), num_classes=1, transform=transform
        )
    else:
        raise RuntimeError(f"Invalid dataset name: {args.dataset}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return train_loader


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    train_loader = check_dataset(args)
    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])

    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    running_avgs = OrderedDict()

    def step(engine, batch):
        x, _ = batch
        x = x.to(device)

        n_batch = len(x)

        optimizer.zero_grad()

        y = transformer(x)

        x = utils.normalize_batch(x)
        y = utils.normalize_batch(y)

        features_x = vgg(x)
        features_y = vgg(y)

        content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.0
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
        style_loss *= args.style_weight

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

        return {"content_loss": content_loss.item(), "style_loss": style_loss.item(), "total_loss": total_loss.item()}

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        args.checkpoint_model_dir, "checkpoint", n_saved=10, require_empty=False, create_dir=True
    )
    progress_bar = Progbar(loader=train_loader, metrics=running_avgs)

    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED(every=args.checkpoint_interval),
        handler=checkpoint_handler,
        to_save={"net": transformer},
    )
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=progress_bar)
    trainer.run(train_loader, max_epochs=args.epochs)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = torch.load(args.model)
        style_model.to(device)
        output = style_model(content_image).cpu()
        utils.save_image(args.output_image, output[0])


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2, help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch_size", type=int, default=8, help="batch size for training, default is 8")
    train_arg_parser.add_argument(
        "--dataset", type=str, required=True, choices={"test", "folder", "mscoco"}, help="type of dataset to be used."
    )
    train_arg_parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="path to training dataset, the path should point to a folder "
        "containing another folder with all the training images",
    )
    train_arg_parser.add_argument("--style_image", type=str, default="test", help="path to style-image")
    train_arg_parser.add_argument("--test_image", type=str, default="test", help="path to test-image")
    train_arg_parser.add_argument(
        "--checkpoint_model_dir",
        type=str,
        default="/tmp/checkpoints",
        help="path to folder where checkpoints of trained models will be saved",
    )
    train_arg_parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="number of batches after which a checkpoint of trained model will be created",
    )
    train_arg_parser.add_argument(
        "--image_size", type=int, default=256, help="size of training images, default is 256 X 256"
    )
    train_arg_parser.add_argument(
        "--style_size", type=int, default=None, help="size of style-image, default is the original size of style image"
    )
    train_arg_parser.add_argument("--cuda", type=int, default=1, help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument(
        "--content_weight", type=float, default=1e5, help="weight for content-loss, default is 1e5"
    )
    train_arg_parser.add_argument(
        "--style_weight", type=float, default=1e10, help="weight for style-loss, default is 1e10"
    )
    train_arg_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate, default is 1e-3")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument(
        "--content_image", type=str, required=True, help="path to content image you want to stylize"
    )
    eval_arg_parser.add_argument(
        "--content_scale", type=float, default=None, help="factor for scaling down the content image"
    )
    eval_arg_parser.add_argument("--output_image", type=str, required=True, help="path for saving the output image")
    eval_arg_parser.add_argument(
        "--model", type=str, required=True, help="saved model to be used for stylizing the image."
    )
    eval_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        raise ValueError("ERROR: specify either train or eval")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    if args.subcommand == "train":
        check_manual_seed(args)
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
