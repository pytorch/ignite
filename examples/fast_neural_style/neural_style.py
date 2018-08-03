# coding: utf-8
from __future__ import print_function, division

import argparse
import os
import sys
import time
import re

import numpy as np
import random
from PIL import Image
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

from collections import OrderedDict

STYLIZED_IMG_FNAME = 'stylized_sample_epoch_{:04d}.png'


def check_paths(args):
    try:
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
        if not os.path.exists(args.stylized_test_dir):
            os.makedirs(args.stylized_test_dir)

    except OSError as e:
        print(e)
        sys.exit(1)


def check_manual_seed(args):
    seed = args.seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True, transform=transform)
    else:
        train_dataset = datasets.ImageFolder(args.dataset, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    if args.style_image != 'random':
        style = utils.load_image(args.style_image, size=args.style_size)
    else:
        imarray = np.random.rand(256, 256, 3).astype('uint8')
        style = Image.fromarray(imarray)

    style = style_transform(style)

    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    running_avgs = OrderedDict()
    output_stream = sys.stdout
    alpha = 0.98

    def step(engine, batch):

        x, _ = batch
        x = x.to(device)

        n_batch = len(x)

        transformer.zero_grad()

        y = transformer(x)

        x = utils.normalize_batch(x)
        y = utils.normalize_batch(y)

        features_x = vgg(x)
        features_y = vgg(y)

        content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
        style_loss *= args.style_weight

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'total_loss': total_loss.item()
        }

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(args.checkpoint_model_dir, 'checkpoint',
                                         save_interval=args.checkpoint_interval,
                                         n_saved=10, require_empty=False, create_dir=True)

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_logs(engine):
        for k, v in engine.state.output.items():
            old_v = running_avgs.get(k, v)
            new_v = alpha * old_v + (1 - alpha) * v
            running_avgs[k] = new_v

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):

        num_seen = engine.state.iteration - len(train_loader) * (engine.state.epoch - 1)

        percent_seen = 100 * (float(num_seen / len(train_loader)))
        percentages = list(range(0, 110, 10))

        if int(percent_seen) == 100:
            progress = 0
            equal_to = 10
            sub = 0

        else:
            sub = 1
            progress = 1
            equal_to = np.max(np.where([percent < percent_seen for percent in percentages])[0])

        bar = '[' + '=' * equal_to + '>' * progress + ' ' * (10 - equal_to - sub) + ']'

        message = 'Epoch {epoch} | {percent_seen:.2f}% | {bar}'.format(epoch=engine.state.epoch,
                                                                       percent_seen=percent_seen,
                                                                       bar=bar)
        for key, value in running_avgs.items():
            message += ' | {name}: {value:.2e}'.format(name=key, value=value)

        message += '\r'

        output_stream.write(message)
        output_stream.flush()

    @trainer.on(Events.EPOCH_COMPLETED)
    def complete_progress(engine):
        output_stream.write('\n')

    @trainer.on(Events.EPOCH_COMPLETED)
    def stylize_image(engine):
        path = os.path.join(args.stylized_test_dir, STYLIZED_IMG_FNAME.format(engine.state.epoch))

        if args.test_image != 'random':
            content_image = utils.load_image(args.test_image, scale=None)
        else:
            imarray = np.random.rand(256, 256, 3).astype('uint8')
            content_image = Image.fromarray(imarray)

        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = transformer(content_image).cpu()

        utils.save_image(path, output[0])

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'net': transformer})
    trainer.run(train_loader, max_epochs=args.epochs)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.content_image != 'random':
        content_image = utils.load_image(args.content_image, scale=args.content_scale)
    else:
        imarray = np.random.rand(256, 256, 3).astype('uint8')
        content_image = Image.fromarray(imarray)

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
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
    train_arg_parser.add_argument("--batch_size", type=int, default=8,
                                  help="batch size for training, default is 32")
    train_arg_parser.add_argument("--dataset", type=str, default='cifar10',
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style_image", type=str, default="random",
                                  help="path to style-image")
    train_arg_parser.add_argument("--test_image", type=str, default="random",
                                  help="path to test-image")
    train_arg_parser.add_argument("--checkpoint_model_dir", type=str, default='/tmp/checkpoints',
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--checkpoint_interval", type=int, default=1,
                                  help="number of batches after which a checkpoint of trained model will be created")
    train_arg_parser.add_argument("--stylized_test_dir", type=str, default='/tmp/images/stylized_test',
                                  help="path to folder where stylized test image will be saved at end of each epoch")
    train_arg_parser.add_argument("--image_size", type=int, default=32,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style_size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, default=1,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content_weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style_weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-4")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content_image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content_scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output_image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image.")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_manual_seed(args)
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == '__main__':
    main()
