import numpy as np
from torch import nn
import torch
from torch.functional import F

class PeopleNumberPredictionModel(nn.Module):

    def __init__(self, out_layer_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 50, 1, stride=2)
        self.conv2 = nn.Conv1d(10, 1, 2)
        self.max_pool = nn.MaxPool1d(2)
        self.full = nn.Linear(50, out_layer_size)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = F.relu(h1)
        h3 = self.max_pool(h2)

        h4 = self.conv2(h3)
        h5 = F.relu(h4)

        return self.full(h5)
