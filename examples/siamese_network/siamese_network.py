import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import LRScheduler


class SiameseNetwork(nn.Module):
    # update Siamese Network implementation in accordance with the dataset
    """
    Siamese network for image similarity estimation.
    The network is composed of two identical networks, one for each input.
    The output of each network is concatenated and passed to a linear layer.
    The output of the linear layer passed through a sigmoid function.
    `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
    This implementation varies from FaceNet as we use the `ResNet-18` model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`
    as our feature extractor.
    In addition we use CIFAR10 dataset along with TripletMarginLoss
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        # initialise sigmoid activation
        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input):

        # pass the input through resnet
        output = self.forward_once(input)

        # pass the output of resnet to sigmoid layer
        output = self.sigmoid(output)

        return output


class MatcherDataset(Dataset):
    # following class implements data downloading and handles preprocessing
    def __init__(self, root, train, download=False):
        super(MatcherDataset, self).__init__()

        # get CIFAR10 dataset
        self.dataset = datasets.CIFAR10(root, train=train, download=download)

        # convert data from numpy array to Tensor
        self.data = torch.from_numpy(self.dataset.data)

        # shift the dimensions of dataset to match the initial input layer dimensions
        self.data = torch.movedim(self.data, (0, 1, 2, 3), (0, 2, 3, 1))

        # convert targets list to torch Tensor
        self.dataset.targets = torch.Tensor(self.dataset.targets)

        self.group_examples()

    def group_examples(self):
        """
        To ease the accessibility of data based on the class, we will use `group_examples` to group
        examples based on class. The data classes have already been mapped to numeric values and
        so are the target outputs for each training input

        Every key in `grouped_examples` corresponds to a class in CIFAR10 dataset. For every key in
        `grouped_examples`, every value will conform to all of the indices for the CIFAR10
        dataset examples that correspond to that key.
        """

        # get the targets from CIFAR10 dataset
        np_arr = np.array(self.dataset.targets.clone())

        # group examples based on class
        self.grouped_examples = {}
        for i in range(0, 10):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        For every sample in the batch we select 3 images. First one is the anchor image
        which is the image obtained from the current index. We also obtain the label of
        anchor image.

        Now we select two random images, one belonging to the same class as that of the
        anchor image (named as positive_image) and the other belonging to a different class
        than that of the anchor image (named as negative_image). We return the anchor image,
        positive image, negative image and anchor label.
        """

        # obtain the anchor image
        anchor_image = self.data[index].clone().float()

        # obtain the class label of the anchor image
        anchor_label = self.dataset.targets[index].clone().float()

        # pick some random class for the first image
        selected_class = random.randint(0, 9)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

        while random_index_1 != anchor_label:
            random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        positive_image = self.data[index_1].clone().float()

        random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

        # ensure that the index of the second image isn't the same as the first image
        while random_index_2 == anchor_label:
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

        # pick the index to get the second image
        index_2 = self.grouped_examples[selected_class][random_index_2]

        # get the second image
        negative_image = self.data[index_2].clone().float()

        return anchor_image, positive_image, negative_image, anchor_label


def train(model, device, optimizer, train_loader, lr_scheduler, log_interval, max_epochs):

    criterion = nn.TripletMarginLoss()

    # define model training step
    def train_step(engine, batch):
        model.train()
        anchor_image, positive_image, negative_image, anchor_label = batch
        anchor_image = anchor_image.to(device)
        positive_image, negative_image = positive_image.to(device), negative_image.to(device)
        anchor_label = anchor_label.to(device)
        optimizer.zero_grad()
        anchor_out = model(anchor_image)
        positive_out = model(positive_image)
        negative_out = model(negative_image)
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        return loss

    # create a trainer engine and attach train_step
    trainer = Engine(train_step)

    # attach progress bar to trainer
    pbar = ProgressBar()
    pbar.attach(trainer)

    # attach various handlers to trainer engine
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_results(engine):
        print(f"Train Epoch: {engine.state.epoch}, Train Loss: {engine.state.output: .5f}")

    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # run trainer engine
    trainer.run(train_loader, max_epochs=max_epochs)


def test(model, device, test_loader, lr_scheduler, log_interval):

    criterion = nn.TripletMarginLoss()
    average_test_loss = 0

    # define model testing step
    def test_step(engine, batch):
        model.eval()
        anchor_image, positive_image, negative_image, anchor_label = batch
        anchor_image = anchor_image.to(device)
        positive_image, negative_image = positive_image.to(device), negative_image.to(device)
        anchor_label = anchor_label.to(device)
        anchor_out = model(anchor_image)
        positive_out = model(positive_image)
        negative_out = model(negative_image)
        test_loss = criterion(anchor_out, positive_out, negative_out)
        return test_loss

    # create evaluator engine and attach test step
    evaluator = Engine(test_step)

    # attach progress bar to evaluator
    pbar = ProgressBar()
    pbar.attach(evaluator)

    # attach various handlers to evaluator engine
    @evaluator.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_testing_results(engine):
        nonlocal average_test_loss
        average_test_loss += engine.state.output
        print(f"Test Epoch: {engine.state.epoch} Test Loss: {engine.state.output: .5f}")

    evaluator.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # run evaluator engine
    evaluator.run(test_loader)

    # print average loss over test dataset
    print(f"Average Test Loss: {average_test_loss/len(test_loader.dataset): .7f}")


def main():
    # adds training defaults and support for terminal arguments
    parser = argparse.ArgumentParser(description="PyTorch Siamese network Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=200, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument(
        "--gamma", type=float, default=0.95, metavar="M", help="Learning rate step gamma (default: 0.7)"
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--no-mps", action="store_true", default=False, help="disables macOS GPU training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # data loading
    train_dataset = MatcherDataset("../data", train=True, download=True)
    test_dataset = MatcherDataset("../data", train=False)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    # set model parameters
    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=args.gamma)
    lr_scheduler = LRScheduler(scheduler)

    # call train function
    train(model, device, optimizer, train_loader, lr_scheduler, log_interval=args.log_interval, max_epochs=args.epochs)

    # call test function
    test(model, device, test_loader, lr_scheduler, log_interval=args.log_interval)


if __name__ == "__main__":
    main()
