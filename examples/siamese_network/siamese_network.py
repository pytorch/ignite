from __future__ import print_function
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets


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
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' feature
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output


class APP_MATCHER():
    # following class implements data downloading and handles preprocessing
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()

        # get MNIST dataset
        self.dataset = datasets.MNIST(root, train=train, download=download)

        # as `self.dataset.data`'s shape is (Nx28x28), where N is the number of
        # examples in MNIST dataset, a single example has the dimensions of
        # (28x28) for (WxH), where W and H are the width and the height of the image.
        # However, every example should have (CxWxH) dimensions where C is the number
        # of channels to be passed to the network. As MNIST contains gray-scale images,
        # we add an additional dimension to corresponds to the number of channels.
        self.data = self.dataset.data.unsqueeze(1).clone()

        self.group_examples()

    def group_examples(self):
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group
            examples based on class.

            Every key in `grouped_examples` corresponds to a class in MNIST dataset. For every key in
            `grouped_examples`, every value will conform to all of the indices for the MNIST
            dataset examples that correspond to that key.
        """

        # get the targets from MNIST dataset
        np_arr = np.array(self.dataset.targets.clone())

        # group examples based on class
        self.grouped_examples = {}
        for i in range(0,10):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases,
            positive and negative examples. For positive examples, we will have two
            images from the same class. For negative examples, we will have two images
            from different classes.

            Given an index, if the index is even, we will pick the second image from the same class,
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn
            the similarity between two different images representing the same class. If the index is odd, we will
            pick the second image from a different class than the first image.
        """

        # pick some random class for the first image
        selected_class = random.randint(0, 9)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        image_1 = self.data[index_1].clone().float()

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, 9)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 9)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0] - 1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target


def train(model, device, optimizer, train_loader, lr_scheduler, log_interval, max_epochs):

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    # define model training step
    def train_step(engine, batch):
        model.train()
        image_1, image_2, target = batch
        image_1, image_2, target = image_1.to(device), image_2.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(image_1, image_2,).squeeze()
        loss = criterion(outputs, target)
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
        print(f'Train Epoch: {engine.state.epoch}, Train Loss: {engine.state.output: .5f}')

    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # run trainer engine
    trainer.run(train_loader, max_epochs=max_epochs)


def test(model, device, test_loader, lr_scheduler, log_interval):

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()
    average_test_loss = 0

    # define model testing step
    def test_step(engine, batch):
        model.eval()
        image_1, image_2, target = batch
        image_1, image_2, target = image_1.to(device), image_2.to(device), target.to(device)
        outputs = model(image_1, image_2).squeeze()
        test_loss = criterion(outputs, target)
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
        print(f'Test Epoch: {engine.state.epoch} Test Loss: {engine.state.output: .5f}')

    evaluator.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # run evaluator engine
    evaluator.run(test_loader)

    # print average loss over test dataset
    print(f'Average Test Loss: {average_test_loss/len(test_loader.dataset): .7f}')


def main():
    # adds training defaults and support for terminal arguments
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # data loading
    train_dataset = APP_MATCHER('../data', train=True, download=True)
    test_dataset = APP_MATCHER('../data', train=False)
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


if __name__ == '__main__':
    main()
