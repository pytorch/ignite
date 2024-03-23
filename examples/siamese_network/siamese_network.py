import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from ignite.engine import Engine, Events

from ignite.handlers import ProgressBar
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy, RunningAverage
from ignite.utils import manual_seed


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
        self.resnet = torchvision.models.resnet34(weights=None)
        fc_in_features = self.resnet.fc.in_features

        # changing the FC layer of resnet model to a linear layer
        self.resnet.fc = nn.Identity()

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
            nn.ReLU(inplace=True),
        )

        # initialise relu activation
        self.relu = nn.ReLU()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2, input3):
        # pass the input through resnet
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        # pass the output of resnet to sigmoid layer
        output1 = self.fc(output1)
        output2 = self.fc(output2)
        output3 = self.fc(output3)

        return output1, output2, output3


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
        self.dataset.targets = torch.tensor(self.dataset.targets)

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
        np_arr = np.array(self.dataset.targets)

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
        anchor_image = self.data[index].float()

        # obtain the class label of the anchor image
        anchor_label = self.dataset.targets[index]
        anchor_label = int(anchor_label.item())

        # find a label which is different from anchor_label
        labels = list(range(0, 10))
        labels.remove(anchor_label)
        neg_index = torch.randint(0, 9, (1,)).item()
        neg_label = labels[neg_index]

        # get a random index from the range range of indices
        random_index = torch.randint(0, len(self.grouped_examples[anchor_label]), (1,)).item()

        # get the index of image in actual data using the anchor label and random index
        positive_index = self.grouped_examples[anchor_label][random_index]

        # choosing a random image using positive_index
        positive_image = self.data[positive_index].float()

        # get a random index from the range range of indices
        random_index = torch.randint(0, len(self.grouped_examples[neg_label]), (1,)).item()

        # get the index of image in actual data using the negative label and random index
        negative_index = self.grouped_examples[neg_label][random_index]

        # choosing a random image using negative_index
        negative_image = self.data[negative_index].float()

        return anchor_image, positive_image, negative_image, anchor_label


def pairwise_distance(input1, input2):
    dist = input1 - input2
    dist = torch.pow(dist, 2)
    return dist


def calculate_loss(input1, input2):
    output = pairwise_distance(input1, input2)
    loss = torch.sum(output, 1)
    loss = torch.sqrt(loss)
    return loss


def run(args, model, device, optimizer, train_loader, test_loader, lr_scheduler):
    # using Triplet Margin Loss
    criterion = nn.TripletMarginLoss(p=2, margin=2.8)

    # define model training step
    def train_step(engine, batch):
        model.train()
        anchor_image, positive_image, negative_image, anchor_label = batch
        anchor_image = anchor_image.to(device)
        positive_image, negative_image = positive_image.to(device), negative_image.to(device)
        anchor_label = anchor_label.to(device)
        optimizer.zero_grad()
        anchor_out, positive_out, negative_out = model(anchor_image, positive_image, negative_image)
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        return loss

    # define model testing step
    def test_step(engine, batch):
        model.eval()
        with torch.no_grad():
            anchor_image, _, _, anchor_label = batch
            anchor_image = anchor_image.to(device)
            anchor_label = anchor_label.to(device)
            other_image = []
            other_label = []
            y_true = []
            for i in range(anchor_image.shape[0]):
                index = torch.randint(0, anchor_image.shape[0], (1,)).item()
                img = anchor_image[index]
                label = anchor_label[index]
                other_image.append(img)
                other_label.append(label)
                if anchor_label[i] == other_label[i]:
                    y_true.append(1)
                else:
                    y_true.append(0)
            other = torch.stack(other_image)
            other_label = torch.tensor(other_label)
            other, other_label = other.to(device), other_label.to(device)
            anchor_out, other_out, _ = model(anchor_image, other, other)
            test_loss = calculate_loss(anchor_out, other_out)
            y_pred = torch.where(test_loss < 3, 1, 0)
            y_true = torch.tensor(y_true)
            return [y_pred, y_true]

    # create engines for trainer and evaluator
    trainer = Engine(train_step)
    evaluator = Engine(test_step)

    # attach Running Average Loss metric to trainer and evaluator engines
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    Accuracy(output_transform=lambda x: x).attach(evaluator, "accuracy")

    # attach progress bar to trainer with loss
    pbar1 = ProgressBar()
    pbar1.attach(trainer, metric_names=["loss"])

    # attach progress bar to evaluator
    pbar2 = ProgressBar()
    pbar2.attach(evaluator)

    # attach LR Scheduler to trainer engine
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # event handler triggers evauator at end of every epoch
    @trainer.on(Events.EPOCH_COMPLETED(every=args.log_interval))
    def test(engine):
        state = evaluator.run(test_loader)
        print(f'Test Accuracy: {state.metrics["accuracy"]}')

    # run the trainer
    trainer.run(train_loader, max_epochs=args.epochs)


def main():
    # adds training defaults and support for terminal arguments
    parser = argparse.ArgumentParser(description="PyTorch Siamese network Example")
    parser.add_argument(
        "--batch-size", type=int, default=256, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=256, metavar="N", help="input batch size for testing (default: 1000)"
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
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    parser.add_argument("--num-workers", default=4, help="number of processes generating parallel batches")
    args = parser.parse_args()

    # set manual seed
    manual_seed(args.seed)

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # data loading
    train_dataset = MatcherDataset("../data", train=True, download=True)
    test_dataset = MatcherDataset("../data", train=False)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)

    # set model parameters
    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=args.gamma)
    lr_scheduler = LRScheduler(scheduler)

    # call run function
    run(args, model, device, optimizer, train_loader, test_loader, lr_scheduler)


if __name__ == "__main__":
    main()
