import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
import numpy as np
import pandas as pd
import os

from model import PeopleNumberPredictionModel
from data_load import TrainDataset

if __name__ == "__main__":

    out_classes = 3
    net = PeopleNumberPredictionModel(out_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Use 80% of data for training and 20% for validation
    number_of_trips = 9
    data = []
    targets = []
    for idx in range(1, number_of_trips):
        data.append(np.loadtxt(f"../processed_data/in_trip_{idx}.txt"))
        targets.append(np.loadtxt(f"../processed_data/out_trip_{idx}.txt"))
    data = np.array(data)
    targets = np.array(targets)

    trainloader = TrainDataset(data, targets)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')