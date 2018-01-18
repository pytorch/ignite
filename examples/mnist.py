from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from ignite.trainer import TrainingEvents, create_supervised
from ignite.handlers import Validate
from ignite.metrics import categorical_accuracy

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
        return F.log_softmax(x, dim=1)


def run():
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root="./.data",
                                    transform=data_transform,
                                    train=True),
                                    batch_size=100,
                                    shuffle=True)

    val_loader = DataLoader(MNIST(download=False,
                                  root="./.data",
                                  transform=data_transform,
                                  train=False),
                                  batch_size=100,
                                  shuffle=False)

    model = Net()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.5)
    trainer = create_supervised(model, optimizer, F.nll_loss, cuda=False)
    trainer.add_event_handler(TrainingEvents.TRAINING_EPOCH_COMPLETED,
                              Validate(val_loader, epoch_interval=1))

    @trainer.on(TrainingEvents.TRAINING_EPOCH_COMPLETED)
    def print_metrics(trainer):
        print(trainer.current_epoch,
              categorical_accuracy(trainer.validation_history))

    trainer.run(train_loader, max_epochs=10)

if __name__ == '__main__':
    run()
