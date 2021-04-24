import numpy as np
import os
import matplotlib.pyplot as plt


def create_directory(path):
    file_idx = 1
    try:
        os.makedirs(path)
    except FileExistsError as err:
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
    print("created directory: {}".format(path))
    return path


def save_result(results, path):
    for key, values in results.items():
        np.savetxt(path + "/{}.csv".format(key), values, encoding="utf-8")


def save_history(history, path):
    for key, values in history.items():
        np.savetxt(path + "/{}_history.csv".format(key), values, encoding="utf-8")
    plot_history(history, path + "/plot.png")


def plot_history(history, path, xlabel='Epoch', ylabel='Loss', marker=None):
    # plt.style.use(['seaborn-colorblind'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    for key, values in history.items():
        ax.plot(np.arange(1, 1 + len(values), 1), np.array(values), marker=marker, label=key)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.clf()
    plt.close()


def plot_roc_curve(fpr, tpr, path="results/roc_curve.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid()
    ax.plot(fpr, tpr)
    plt.title("ROC Curve")
    fig.savefig(path)