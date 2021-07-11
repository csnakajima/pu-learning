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

    def save(self, key, path, index=None):
        if index is None:
            np.savetxt(os.path.join(path, "{}.csv".format(key)), np.array(self.map[key]), encoding="utf-8")
        else:
            np.savetxt(os.path.join(path, "{}-{}.csv".format(key, index)), np.array(self.map[key]), encoding="utf-8")

    def saveall(self, path, index=None):
        for key in self.map.keys():
            if len(self.map[key]) > 0:
                self.save(key, path, index)

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


def save_model(model, path):
    torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth.tar"))


def save_train_history(path, model, history):
    history.saveall(path)
    history.plotall(path)
    save_model(model, path)


def save_test_history(path, histories):
    for i, history in enumerate(histories):
        history.saveall(path, i)


def output_train_results(path, history, prior):
    train_loss, validation_loss = history.get("train_loss"), history.get("validation_loss")
    with open(path, mode='a', encoding="utf-8") as f:
        print("--Training result--\n", file=f)
        print("Train loss : {:.9f}".format(train_loss), file=f)
        print("Validation loss : {:.9f}".format(validation_loss), file=f)
        print("Training Prior : {:.6f}".format(prior), file=f)
        print("", file=f)


def output_test_results(path, test_idx, true_prior, acc, auc, prior=None, thresh=None, boundary=None):
    with open(path, mode='a', encoding="utf-8") as f:
        if test_idx == 0:
            print("--Test result---\n", file=f)
        print("Test {} : Dataset prior = {}".format(test_idx, true_prior), file=f)
        print("Accuracy : {:.6f}".format(acc), file=f)
        print("AUC : {:.6f}".format(auc), file=f)
        if prior is not None:
            print("Prior : {:.6f}".format(prior), file=f)
        if thresh is not None:
            print("Thresh : {:.6f}".format(thresh), file=f)
        if boundary is not None:
            print("Boundary : {:.6f}".format(boundary), file=f)
        print("", file=f)


def append_test_results(path, acc, auc, prior=None, thresh=None, boundary=None):
    with open(os.path.join(path, "accuracy.txt"), mode='a', encoding="utf-8") as f:
        print(acc, file=f)
    with open(os.path.join(path, "auc.txt"), mode='a', encoding="utf-8") as f:
        print(auc, file=f)
    if prior is not None:
        with open(os.path.join(path, "prior.txt"), mode='a', encoding="utf-8") as f:
            print(prior, file=f)
    if thresh is not None:
        with open(os.path.join(path, "thresh.txt"), mode='a', encoding="utf-8") as f:
            print(thresh, file=f)
    if boundary is not None:
        with open(os.path.join(path, "boundary.txt"), mode='a', encoding="utf-8") as f:
            print(boundary, file=f)


def output_config(path, train_size, val_size, max_epochs, batch_size, lr, alpha, seed):
    with open(path, mode='a', encoding="utf-8") as f:
        print("--Parameters--", file=f)
        print("train_size = {}".format(train_size), file=f)
        print("validation_size = {}".format(val_size), file=f)
        print("max_epochs = {}".format(max_epochs), file=f)
        print("batch_size = {}".format(batch_size), file=f)
        print("lr = {}".format(lr), file=f)
        print("alpha = {}".format(alpha), file=f)
        print("random seed = {}".format(seed), file=f)


def getdirs(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

