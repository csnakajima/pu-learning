import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


class PUDataset(torch.utils.data.Dataset):
    def __init__(self, pos_labels, num_labeled=0, prior=None, indices=None, train=True):
        if indices is not None:
            assert np.max(indices) < self.__len__()
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]
        # prior shift
        if prior is not None:
            assert 0 < prior < 1
            self.prior_shift(pos_labels, prior)
        # test set
        if train is False:
            self.target_transform = lambda x: 1 if x in pos_labels else -1
        # positive set
        elif num_labeled > 0:
            self.assign_positive(pos_labels, num_labeled)
            self.target_transform = lambda x: 1
        # unlabeled set
        else:
            self.target_transform = lambda x: -1

    def assign_positive(self, pos_labels, num_labeled):
        isPositive = np.vectorize(lambda x: x in pos_labels)
        indices = np.where(isPositive(self.targets) == True)[0]
        assert num_labeled <= len(indices)
        np.random.shuffle(indices)
        indices = indices[:num_labeled]
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def prior_shift(self, pos_labels, prior):
        isPositive = np.vectorize(lambda x: x in pos_labels)
        pos_indices = np.where(isPositive(self.targets) == True)[0]
        neg_indices = np.where(isPositive(self.targets) == False)[0]
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)
        prior_of_dataset = len(pos_indices) / self.__len__()
        if prior < prior_of_dataset:
            pos_indices = pos_indices[:int(prior / (1 - prior) * len(neg_indices))]
        else:
            neg_indices = neg_indices[:int((1 - prior) / prior * len(pos_indices))]
        indices = np.concatenate([pos_indices, neg_indices])
        self.data = self.data[indices]
        self.targets = self.targets[indices]


class MNIST(torchvision.datasets.MNIST, PUDataset):
    def __init__(self, root, pos_labels, num_labeled=0, prior=None, indices=None, train=True, transform=None, target_transform=None, download=True):
        torchvision.datasets.MNIST.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        PUDataset.__init__(self, pos_labels, num_labeled, prior, indices, train)


class FashionMNIST(torchvision.datasets.FashionMNIST, PUDataset):
    def __init__(self, root, pos_labels, num_labeled=0, prior=None, indices=None, train=True, transform=None, target_transform=None, download=True):
        torchvision.datasets.FashionMNIST.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        PUDataset.__init__(self, pos_labels, num_labeled, prior, indices, train)


class CIFAR10(torchvision.datasets.CIFAR10, PUDataset):
    def __init__(self, root, pos_labels, num_labeled=0, prior=None, indices=None, train=True, transform=None, target_transform=None, download=True):
        torchvision.datasets.CIFAR10.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        PUDataset.__init__(self, pos_labels, num_labeled, prior, indices, train)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num, prior, transform=None, target_transform=None):
        self.prior = prior
        self.transform = transform
        self.target_transform = target_transform
        self.rng = np.random.default_rng()
        self.generate(num)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        instance = self.data[idx] if self.transform is None else self.transform(self.data[idx])
        label = self.targets[idx] if self.target_transform is None else self.target_transform(self.targets[idx])
        return instance, label

    def generate(self, num):
        pass


class Gaussian(SyntheticDataset):
    def random_positives(self, num=1):
        mean = [1, 1]
        var = [[1, 0], [0, 1]]
        return self.rng.multivariate_normal(mean, var, size=num, check_valid="raise")

    def random_negatives(self, num=1):
        mean = [-1, -1]
        var = [[1, 0], [0, 1],]
        return self.rng.multivariate_normal(mean, var, size=num, check_valid="raise")

    def generate(self, num):
        num_P = int(num * self.prior)
        num_N = num - num_P
        positives = self.random_positives(num_P)
        negatives = self.random_negatives(num_N)
        self.data = np.concatenate([positives, negatives])
        self.targets = np.array([1] * num_P + [-1] * num_N)


class ndarray_to_Tensor(object):
    def __init__(self):
        pass

    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        return torch.from_numpy(x).float()


def select_transform(dataset_name):
    transforms = {
        "mnist": torchvision.transforms.ToTensor(),
        "fmnist": torchvision.transforms.ToTensor(),
        "cifar": torchvision.transforms.ToTensor()
    }
    return transforms[dataset_name]


# num_labeled > 0 -> Positive set, num_labeled = 0 -> Unlabeled set
def get_dataset(dataset_name, pos_labels, num_labeled=0, prior=None, indices=None, train=True):
    datasets = {
        "mnist": MNIST,
        "fmnist": FashionMNIST,
        "cifar": CIFAR10 
    }
    return datasets[dataset_name]("dataset", pos_labels, num_labeled, prior, indices, train, transform=select_transform(dataset_name))
