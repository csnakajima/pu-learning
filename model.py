import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnivariateClassifier(nn.Module):
    def __init__(self, activate_output=False, dim=1):
        super(UnivariateClassifier, self).__init__()
        self.l1 = nn.Linear(dim, 1, bias=True)
        self.l2 = nn.Linear(1, 1, bias=False)
        # self.l2.weight.data.fill_(0)
        self.af = nn.Sigmoid()
        self.activate_output = activate_output

    def forward(self, x):
        x = self.l1(x)
        if self.activate_output:
            x = self.l2(self.af(x))
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, dim):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(dim, 300, bias=False)
        self.b1 = nn.BatchNorm1d(300)
        self.l2 = nn.Linear(300, 300, bias=False)
        self.b2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 1)
        self.af = nn.ReLU()

    def forward(self, x):
        h = self.l1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.l2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.l3(h)
        return h


class CNN_MNIST(nn.Module):
    def __init__(self, activate_output=False):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.b1 = nn.BatchNorm2d(6)
        self.b2 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(120 * 1 * 1, 100)
        self.b3 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 1)
        self.af = nn.ReLU()
        self.activate_output = activate_output

    def forward(self, x):
        in_size = x.size(0)
        x = self.af(self.mp(self.conv1(x)))
        x = self.af(self.mp(self.conv2(x)))
        x = self.af(self.conv3(x))
        x = x.view(in_size, -1)
        x = self.af(self.b3(self.fc1(x)))
        x = self.fc2(x)
        if self.activate_output:
            x = self.af(x)
        return x


class CNN_CIFAR(nn.Module):
    def __init__(self, activate_output=False):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=2)
        self.conv3 = nn.Conv2d(96, 192, 1)
        self.conv4 = nn.Conv2d(192, 10, 1)
        self.fc1 = nn.Linear(1960, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2)
        self.af = F.relu

    def forward(self, x):
        x = self.af(self.conv1(x))
        x = self.af(self.conv2(x))
        x = self.af(self.conv3(x))
        x = self.af(self.conv4(x))
        x = x.view(-1, 1960)
        x = self.af(self.fc1(x))
        x = self.af(self.fc2(x))
        x = self.fc3(x)
        if self.activate_output:
            x = self.af(x)
        return x


def select_model(model_name, activate_output=False):
    models = {
        "gauss": UnivariateClassifier,
        "gauss_mix": UnivariateClassifier,
        "mnist": CNN_MNIST,
        "fmnist": CNN_MNIST,
        "cifar": CNN_CIFAR
    }
    return models[model_name]
