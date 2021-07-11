import numpy as np
import torch
import torchvision


class ImageDataset(torch.utils.data.Dataset):
    pos_labels = []
    
    def __init__(self, num_labeled=0, prior=None, indices=None, train=True):
        self.dataset_type = "Image"
        if indices is not None:
            assert np.max(indices) < self.__len__()
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]
        else:
            self.targets = np.array(self.targets)
        # prior shift
        if prior is not None:
            assert 0 < prior < 1
            self.prior_shift(prior)
        if num_labeled > 0:
            self.assign_positive(num_labeled)
        if train is False:
            self.data = self.data[:5000]
            self.targets = self.targets[:5000]

    def assign_positive(self, num_labeled):
        isPositive = np.vectorize(lambda x: x in ImageDataset.pos_labels)
        indices = np.where(isPositive(self.targets) == True)[0]
        assert num_labeled <= len(indices)
        np.random.shuffle(indices)
        indices = indices[:num_labeled]
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]

    def prior_shift(self, prior):
        isPositive = np.vectorize(lambda x: x in ImageDataset.pos_labels)
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
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.targets = self.targets[indices]


class MNIST(torchvision.datasets.MNIST, ImageDataset):
    def __init__(self, root, num_labeled=0, prior=None, indices=None, train=True, transform=None, target_transform=None, download=True):
        torchvision.datasets.MNIST.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        ImageDataset.__init__(self, num_labeled, prior, indices, train)


class FashionMNIST(torchvision.datasets.FashionMNIST, ImageDataset):
    def __init__(self, root, num_labeled=0, prior=None, indices=None, train=True, transform=None, target_transform=None, download=True):
        torchvision.datasets.FashionMNIST.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        ImageDataset.__init__(self, num_labeled, prior, indices, train)


class KMNIST(torchvision.datasets.KMNIST, ImageDataset):
    def __init__(self, root, num_labeled=0, prior=None, indices=None, train=True, transform=None, target_transform=None, download=True):
        torchvision.datasets.KMNIST.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        ImageDataset.__init__(self, num_labeled, prior, indices, train)


class CIFAR10(torchvision.datasets.CIFAR10, ImageDataset):
    def __init__(self, root, num_labeled=0, prior=None, indices=None, train=True, transform=None, target_transform=None, download=True):
        torchvision.datasets.CIFAR10.__init__(self, root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        ImageDataset.__init__(self, num_labeled, prior, indices, train)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, num, prior, transform=None, target_transform=None):
        self.prior = prior
        self.transform = transform
        self.target_transform = target_transform
        # self.rng = np.random.default_rng()
        self.generate(num)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        instance = self.data[idx] if self.transform is None else self.transform(self.data[idx])
        label = self.targets[idx] if self.target_transform is None else self.target_transform(self.targets[idx])
        return instance, label

    def min_max(self):
        return (self.data.min(), self.data.max())

    def generate(self, num):
        num_P = int(num * self.prior)
        num_N = num - num_P
        positives = self.random_positives(num_P)
        negatives = self.random_negatives(num_N)
        self.data = np.concatenate([positives, negatives])
        self.targets = np.array([1] * num_P + [-1] * num_N)

    def base1(self, num=1):
        mean = 1
        var = 1
        return np.random.normal(mean, var, (num, 1))

    def base2(self, num=1):
        mean = -1
        var = 1
        return np.random.normal(mean, var, (num, 1))


class Gaussian(SyntheticDataset):
    def random_positives(self, num):
        theta = 1
        return np.concatenate([self.base1(int(num * theta)), self.base2(num - int(num * theta))])

    def random_negatives(self, num):
        theta = 0
        return np.concatenate([self.base1(int(num * theta)), self.base2(num - int(num * theta))])


class Gaussian_Mixture(SyntheticDataset):
    def random_positives(self, num):
        theta = 0.8
        return np.concatenate([self.base1(int(num * theta)), self.base2(num - int(num * theta))])

    def random_negatives(self, num):
        theta = 0.2
        return np.concatenate([self.base1(int(num * theta)), self.base2(num - int(num * theta))])


class ndarray_to_Tensor(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.from_numpy(x).float()


class to_1dTensor(object):
    def __call__(self, x):
        return x.view(-1)


class Tensor_to_1darray(object):
    def __init__(self):
        pass

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return x
        else:
            return x.view(-1).detach().numpy().copy()


def choose_image_dataset(dataset_name):
    datasets = {
        "mnist": MNIST,
        "fmnist": FashionMNIST,
        "kmnist": KMNIST,
        "cifar": CIFAR10
    }
    return datasets[dataset_name]


def choose_synthetic_dataset(dataset_name):
    datasets = {
        "gauss": Gaussian,
        "gauss_mix": Gaussian_Mixture
    }
    return datasets[dataset_name]


def get_image_positive(dataset_name, num_labeled, indices=None, root="dataset"):
    transform = torchvision.transforms.ToTensor()
    target_transform = torchvision.transforms.Lambda(lambda x: 1)
    return choose_image_dataset(dataset_name)(root, num_labeled, indices=indices, train=True, transform=transform, target_transform=target_transform)


def get_image_unlabeled(dataset_name, indices=None, root="dataset"):
    transform = torchvision.transforms.ToTensor()
    target_transform = torchvision.transforms.Lambda(lambda x: -1)
    return choose_image_dataset(dataset_name)(root, indices=indices, train=True, transform=transform, target_transform=target_transform)


def get_image_test(dataset_name, prior, indices=None, root="dataset"):
    transform = torchvision.transforms.ToTensor()
    target_transform = torchvision.transforms.Lambda(lambda x: 1 if x in ImageDataset.pos_labels else -1)
    return choose_image_dataset(dataset_name)(root, prior=prior, indices=indices, train=False, transform=transform, target_transform=target_transform)


def get_synthetic_positive(dataset_name, num):
    transform = ndarray_to_Tensor()
    target_transform = torchvision.transforms.Lambda(lambda x: 1)
    return choose_synthetic_dataset(dataset_name)(num, prior=1, transform=transform, target_transform=target_transform)


def get_synthetic_unlabeled(dataset_name, num, prior):
    transform = ndarray_to_Tensor()
    target_transform = torchvision.transforms.Lambda(lambda x: -1)
    return choose_synthetic_dataset(dataset_name)(num, prior, transform=transform, target_transform=target_transform)


def get_synthetic_test(dataset_name, num, prior):
    transform = ndarray_to_Tensor()
    target_transform = None
    return choose_synthetic_dataset(dataset_name)(num, prior, transform=transform, target_transform=target_transform)

