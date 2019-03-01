import os
import tempfile
import shutil

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite.contrib.handlers import *


@pytest.fixture
def dirname():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


def test_log_graph(dirname):

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

    tb_logger = TensorboardLogger(log_dir=dirname)

    model = Net()
    x = torch.rand(2, 1, 28, 28)
    tb_logger.log_graph(model, x)
    tb_logger = None

    files = [f for f in os.listdir(dirname)]
    assert len(files) >= 1 and "events.out.tfevents" in files[0]
