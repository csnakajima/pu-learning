import numpy as np
import torch
import os
import matplotlib.pyplot as plt


class Results(object):
    def __init__(self, keys):
        self.map = dict()
        for k in keys:
            self.map[k] = []
        
    def append(self, key, value):
        self.map[key].append(value)

    def get(self, key, index=-1):
        return self.map.get(key, [0])[index]

    def empty(self, key):
        return len(self.map[key]) == 0

    def mean(self, key):
        return np.array(self.map[key]).mean()

    def stdev(self, key):
        return np.array(self.map[key]).std()

    def save(self, key, path):
        np.savetxt(path + "/{}.csv".format(key), np.array(self.map[key]), encoding="utf-8")

    def saveall(self, path):
        for key in self.map.keys():
            if len(self.map[key]) > 0:
                self.save(key, path)

    def plot(self, keys, path, xlabel='Epoch', ylabel='Loss', marker=None):
        # plt.style.use(['seaborn-colorblind'])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()
        for key in keys:
            ax.plot(np.arange(1, 1 + len(self.map[key]), 1), np.array(self.map[key]), marker=marker, label=key)
        ax.legend(loc="best")
        fig.savefig(path)
        plt.clf()
        plt.close()

    def plotall(self, path, xlabel="Epoch", ylabel="Loss", marker=None):
        self.plot(self.map.keys(), path, xlabel, ylabel, marker)


def create_directory(path, another_directory=True):
    file_idx = 1
    try:
        os.makedirs(path)
    except FileExistsError as err:
        if another_directory is True:
            print("catch FileExistError: ", err)
            file_idx = 1
            while file_idx < 10:
                try:
                    os.makedirs(path + "_{}".format(file_idx))
                except FileExistsError as err:
                    print("catch FileExistError: ", err)
                    file_idx += 1
                else:
                    path = path + "_{}".format(file_idx)
                    break
            assert file_idx < 10
    print("Save to directory {}".format(path))
    return path


def save_model(model, path):
    torch.save(model.state_dict(), path + "/checkpoint.pth.tar")


def plot_roc_curve(fpr, tpr, path="results/roc_curve.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid()
    ax.plot(fpr, tpr)
    plt.title("ROC Curve")
    fig.savefig(path)
