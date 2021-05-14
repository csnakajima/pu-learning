import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnivariateNet(nn.Module):
    def __init__(self, activate_output=False, dim=1):
        super(UnivariateNet, self).__init__()
        self.l1 = nn.Linear(dim, 4, bias=True)
        self.l2 = nn.Linear(4, 1, bias=True)
        self.af1 = nn.Sigmoid()
        self.af2 = nn.Softplus()
        self.activate_output = activate_output

    def forward(self, x):
        x = self.af1(self.l1(x))
        x = self.l2(x)
        if self.activate_output:
            x = self.af2(x)
        return x


class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim=784, activate_output=False):
        super(MultiLayerPerceptron, self).__init__()
        self.l1 = nn.Linear(dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, 1)
        self.b1 = nn.BatchNorm1d(300)
        self.b2 = nn.BatchNorm1d(300)
        self.b3 = nn.BatchNorm1d(300)
        self.af = nn.ReLU()
        self.activate_output = activate_output

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.af(self.b1(self.l1(x)))
        x = self.af(self.b2(self.l2(x)))
        x = self.af(self.b3(self.l3(x)))
        x = self.l4(x)
        if self.activate_output:
            x = self.af(x)
        return x


class LeNet(nn.Module):
    def __init__(self, activate_output=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.b1 = nn.BatchNorm2d(6)
        self.b2 = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(120 * 1 * 1, 84)
        self.fc2 = nn.Linear(84, 1)
        self.b3 = nn.BatchNorm1d(84)
        self.af = nn.ReLU()
        self.activate_output = activate_output

    def forward(self, x):
        in_size = x.size(0)
        x = self.af(self.mp1(self.conv1(x)))
        x = self.af(self.mp2(self.conv2(x)))
        x = self.af(self.conv3(x))
        x = x.view(in_size, -1)
        x = self.af(self.b3(self.fc1(x)))
        x = self.fc2(x)
        if self.activate_output:
            x = self.af(x)
        return x


class AllConvNet(nn.Module):
    def __init__(self, activate_output=False):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(96)
        self.b3 = nn.BatchNorm2d(96)
        self.b4 = nn.BatchNorm2d(192)
        self.b5 = nn.BatchNorm2d(192)
        self.b6 = nn.BatchNorm2d(192)
        self.b7 = nn.BatchNorm2d(192)
        self.b8 = nn.BatchNorm2d(192)
        self.b9 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.af = nn.ReLU()
        self.activate_output = activate_output

    def forward(self, x):
        in_size = x.size(0)
        x = self.af(self.b1(self.conv1(x)))
        x = self.af(self.b2(self.conv2(x)))
        x = self.af(self.b3(self.conv3(x)))
        x = self.af(self.b4(self.conv4(x)))
        x = self.af(self.b5(self.conv5(x)))
        x = self.af(self.b6(self.conv6(x)))
        x = self.af(self.b7(self.conv7(x)))
        x = self.af(self.b8(self.conv8(x)))
        x = self.af(self.b9(self.conv9(x)))
        x = x.view(in_size, -1)
        x = self.af(self.fc1(x))
        x = self.af(self.fc2(x))
        x = self.fc3(x)
        if self.activate_output:
            x = self.af(x)
        return x


class ConvPoolNet(nn.Module):
    def __init__(self, activate_output=False):
        super(ConvPoolNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(192)
        self.b3 = nn.BatchNorm2d(192)
        self.b4 = nn.BatchNorm2d(192)
        self.b5 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(8 * 8 * 10, 100)
        self.fc2 = nn.Linear(100, 1)
        self.af = nn.ReLU()
        self.activate_output = activate_output

    def forward(self, x):
        in_size = x.size(0)
        x = self.af(self.b1(self.mp1(self.conv1(x))))
        x = self.af(self.b2(self.mp2(self.conv2(x))))
        x = self.af(self.b3(self.conv3(x)))
        x = self.af(self.b4(self.conv4(x)))
        x = self.af(self.b5(self.conv5(x)))
        x = x.view(in_size, -1)
        x = self.af(self.fc1(x))
        x = self.fc2(x)
        if self.activate_output:
            x = self.af(x)
        return x


def select_model(model_name):
    models = {
        "gauss": UnivariateNet,
        "gauss_mix": UnivariateNet,
        "mnist": MultiLayerPerceptron,
        "fmnist": LeNet,
        "cifar": ConvPoolNet
    }
    return models[model_name]
