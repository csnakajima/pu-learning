import numpy as np
import torch
import torch.nn as nn
import statistics as stats
import os

from torch.nn import functional as F
from matplotlib import pyplot as plt
from dataset import get_dataset
from model import select_model
from metric import *
from save import *
from sklearn.metrics import roc_auc_score, roc_curve


def list_mean(l):
    return np.mean(np.array(l))

def list_stdev(l):
    return np.std(np.array(l))


def to_ndarray(tensor):
    return tensor.to('cpu').detach().numpy().copy()


def PUsequence(num_P, num_U):
    return np.array([1] * num_P + [-1] * num_U)


# y, t : ndarray
def priorestimator(y, t):
    num_P = np.count_nonzero(t == 1)
    delta = 0.01
    eps = np.sqrt(4 * np.log(np.exp(1) * num_P / 2) / num_P) + np.sqrt(np.log(2 / delta) / (2 * num_P))
    fpr, tpr, thresh = roc_curve(t, y)
    idx = tpr > 2 * eps
    ratio = fpr[idx] / tpr[idx]
    ratio = ratio[~(np.isnan(ratio) + np.isinf(ratio))]
    assert np.size(ratio) > 0
    return np.min(ratio)


# get train / validation indices
def random_split(train_size, val_size):
    indices = np.arange(train_size + val_size)
    np.random.shuffle(indices)
    return indices[:train_size], indices[-val_size:]


# load dataset for training / validation
def load_trainset(dataset_name, pos_labels, train_size, val_size, prior=None):
    train_indices, val_indices = random_split(train_size[1], val_size[1])
    train_P = get_dataset(dataset_name, pos_labels, num_labeled=train_size[0], indices=train_indices, train=True)
    train_U = get_dataset(dataset_name, pos_labels, indices=train_indices, train=True)
    val_P = get_dataset(dataset_name, pos_labels, num_labeled=val_size[0], indices=val_indices, train=True)
    val_U = get_dataset(dataset_name, pos_labels, indices=val_indices, train=True)
    return train_P, train_U, val_P, val_U


# load dataset for test
def load_testset(dataset_name, pos_labels, prior=None):
    testset = get_dataset(dataset_name, pos_labels, prior=prior, train=False)
    return testset


# training procedure
def train(device, method, datasets, loss_name, alpha, max_epochs, batch_size, stepsize, max_batch_size=50000):
    # data setup ----------------
    trainset_P, trainset_U, valset_P, valset_U = datasets
    batch_num = len(trainset_U) // batch_size
    trainloader_P = torch.utils.data.DataLoader(trainset_P, batch_size=len(trainset_P)//batch_num, shuffle=True, drop_last=True, num_workers=2)
    trainloader_U = torch.utils.data.DataLoader(trainset_U, batch_size=len(trainset_U)//batch_num, shuffle=True, drop_last=True, num_workers=2)
    valloader_P = torch.utils.data.DataLoader(valset_P, batch_size=min(len(valset_P), max_batch_size), shuffle=False, drop_last=False, num_workers=1)
    valloader_U = torch.utils.data.DataLoader(valset_U, batch_size=min(len(valset_U), max_batch_size), shuffle=False, drop_last=False, num_workers=1)

    # model setup ----------------
    activate_output = False if method in ["uPU", "nnPU"] else True
    model = select_model("CNN")(activate_output).to(device)
    if method == "uPU":
        criterion = PURiskEstimator(prior=alpha, loss=select_loss(loss_name))
        criterion_val = PURiskEstimator(prior=alpha, loss=select_loss("zero-one"))
    elif method == "nnPU":
        criterion = NonNegativeRiskEstimator(prior=alpha, loss=select_loss(loss_name))
        criterion_val = PURiskEstimator(prior=alpha, loss=select_loss("zero-one"))
    else:
        criterion = NonNegativeBregmanDivergence(prior=alpha, f_df=select_loss(loss_name))
        criterion_val = BregmanDivergence(f_df=select_loss(loss_name))

    # trainer setup ----------------
    # stepsize = 1e-5
    optimizer = torch.optim.Adam(params=model.parameters(), lr=stepsize, weight_decay=0.005, betas=(0.9, 0.999))
    history = {"train_loss": list(), "validation_loss": list()}

    for ep in range(max_epochs):
        # train step
        model.train()
        train_loss = []
        for (x_p, t_p), (x_u, t_u) in zip(trainloader_P, trainloader_U):
            num_P, num_U = len(x_p), len(x_u)
            x = torch.cat([x_p, x_u]).to(device)
            y = model(x).view(-1)
            y_p, y_u = y[:num_P], y[num_P:]
            loss = criterion(y_p, y_u)
            loss.backward()
            optimizer.step()
            train_loss.append(criterion.value())
        history["train_loss"].append(list_mean(train_loss))

        # validation step
        model.eval()
        with torch.no_grad():
            validation_loss = []
            for (x_p, t_p), (x_u, t_u) in zip(valloader_P, valloader_U):
                num_P, num_U = len(x_p), len(x_u)
                x = torch.cat([x_p, x_u]).to(device)
                y = model(x).view(-1)
                y_p, y_u = y[:num_P], y[num_P:]
                criterion_val(y_p, y_u)
                validation_loss.append(criterion_val.value())
            history["validation_loss"].append(list_mean(validation_loss))

    if method in ["uPU", "nnPU"]:
        train_prior, preds_P = None, None
    else:
        # estimate class prior
        with torch.no_grad():
            preds_P, preds_U = [], []
            num_P, num_U = len(valset_P), len(valset_U)
            for x_p, t_p in valloader_P:
                x = x_p.to(device)
                y = model(x).view(-1)
                preds_P.append(to_ndarray(y))
            for x_u, t_u in valloader_U:
                x = x_u.to(device)
                y = model(x).view(-1)
                preds_U.append(to_ndarray(y))
            preds_P = np.concatenate(preds_P)
            preds_U = np.concatenate(preds_U)
            train_prior = priorestimator(np.concatenate([preds_P, preds_U]), PUsequence(num_P, num_U))
    
    return model, train_prior, preds_P, history


# test procedure
def test(device, model, testset, train_prior, preds_P, max_batch_size=50000):
    batch_size = min(len(testset), max_batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    accuracy = Accuracy()
    auroc = AUROC()
    preds_U, targets = [], []
    model.eval()
    with torch.no_grad():
        for data, target in testloader:
            x, t = data.to(device), target.to(device)
            y = model(x).view(-1)
            preds_U.append(to_ndarray(y))
            targets.append(to_ndarray(t))
        preds_U = np.concatenate(preds_U)
        targets = np.concatenate(targets)
        # uPU / nnPU
        if preds_P is None:
            test_prior, thresh = 0, 0
        # DRPU
        else:
            test_prior = priorestimator(np.concatenate([preds_P, preds_U]), PUsequence(len(preds_P), len(preds_U)))
            thresh = train_prior * (1 - test_prior) / ((1 - train_prior) * test_prior + train_prior * (1 - test_prior))
            thresh /= train_prior
        acc = accuracy(preds_U - thresh, targets)
        auc = auroc(preds_U, targets)
    return acc, auc, test_prior, thresh


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


def run(device_num, trial, method, dataset_name, loss_name, alpha, pos_labels, priors, train_size, validation_size, max_epochs, batch_size, stepsize, path):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    print("Device : {}".format(device))
    num_test = len(priors)
    home_directory = create_directory("{}/{}-{}-{}".format(path, method, dataset_name, loss_name))
    train_results = {"train_loss": [], "validation_loss": []}
    test_results = [{"accuracy": [], "auc": [], "prior": [], "thresh": []} for i in range(num_test)]

    for trial_idx in range(trial):
        print("Trial {} started.".format(trial_idx + 1))
        # loading training dataset
        train_P, train_U, val_P, val_U = load_trainset(dataset_name, pos_labels, train_size, validation_size)
        # training
        print("Training started.")
        model, train_prior, preds_P, history = train(
            device=device,
            method=method,
            datasets=(train_P, train_U, val_P, val_U),
            loss_name=loss_name,
            alpha=alpha,
            max_epochs=max_epochs,
            batch_size=batch_size,
            stepsize=stepsize
        )
        # save training results
        history_path = create_directory(home_directory + "/train/history-{}".format(trial_idx + 1))
        save_history(history, history_path)
        train_results["train_loss"].append(history["train_loss"][-1])
        train_results["validation_loss"].append(history["validation_loss"][-1])
        # loading test dataset
        testsets = [load_testset(dataset_name, pos_labels, prior=priors[test_idx]) for test_idx in range(num_test)]
        # test
        for test_idx in range(num_test):
            acc, auc, pri, ths = test(
                device=device,
                model=model,
                testset=testsets[test_idx],
                train_prior=train_prior,
                preds_P=preds_P,
            )
            test_results[test_idx]["accuracy"].append(acc)
            test_results[test_idx]["auc"].append(auc)
            test_results[test_idx]["prior"].append(pri)
            test_results[test_idx]["thresh"].append(ths)
        
    # output results
    train_path = home_directory + "/train"
    save_result(train_results, train_path)
    for test_idx in range(num_test):
        test_path = create_directory(home_directory + "/test-{}".format(test_idx + 1))
        save_result(test_results[test_idx], test_path)

    # output summary
    with open(home_directory + "/summary.txt", mode='w', encoding="utf-8") as f:
        print("Trial : {}\nMethod : {}\nDataset : {}\nLoss function : {}\nAlpha : {}\n".format(trial, method, dataset_name, loss_name, alpha), file=f)
        print("--Training result--\n", file=f)
        print("Train loss : {:.9f}".format(list_mean(train_results["train_loss"])), file=f)
        print("Validation loss : {:.9f}\n".format(list_mean(train_results["validation_loss"])), file=f)
        print("--Test result---\n", file=f)
        for test_idx in range(num_test):
            print("Test {} : prior = {}".format(test_idx + 1, priors[test_idx]), file=f)
            print("Accuracy : {:.4f} +/- {:.4f}".format(list_mean(test_results[test_idx]["accuracy"]), list_stdev(test_results[test_idx]["accuracy"])), file=f)
            print("AUC : {:.4f} +/- {:.4f}\n".format(list_mean(test_results[test_idx]["auc"]), list_stdev(test_results[test_idx]["auc"])), file=f)
    
    return list_mean(train_results["validation_loss"])