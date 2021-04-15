import torch


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index][0]

    def __len__(self):
        return len(self.dataset)
