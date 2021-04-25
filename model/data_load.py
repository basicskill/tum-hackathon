import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
import pandas as pd
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, data, target):
        self.X = [torch.tensor(row).float() for row in data]
        self.y = [torch.tensor(row).float() for row in target]

    def __len__(self):
        assert(len(self.X) == len(self.y))
        return len(self.X)

    def __getitem__(self, idx):
        inp = self.X[idx]
        target = self.y[idx]

        sample = [inp, target]
        return sample