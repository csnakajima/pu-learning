import numpy as np
from numpy.lib.scimath import log
import torch
import torch.nn as nn
import torchvision
import os

from torch.nn import functional as F
from matplotlib import pyplot as plt
from dataset import get_image_dataset, get_synthetic_dataset, Tensor_to_1darray
from model import select_model
from metric import *
from save import *
from Kernel_MPE import KM2_estimate
from sklearn.metrics import roc_auc_score, roc_curve

SYNTHETIC = ["gauss", "gauss_mix"]
UNBIASED = ["uPU", "nnPU"]

def to_ndarray(tensor):
    return tensor.to('cpu').detach().numpy().copy()


def PUsequence(num_P, num_U):
    return np.array([1] * num_P + [-1] * num_U)


# y, t : ndarray
def priorestimator(y, t):
    num_P = np.count_nonzero(t == 1)
    delta = 1 / num_P
    eps = np.sqrt(4 * np.log(np.exp(1) * num_P / 2) / num_P) + np.sqrt(np.log(2 / delta) / (2 * num_P))
    fpr, tpr, thresh = roc_curve(t, y)
    idx = tpr > eps
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
def load_trainset(dataset_name, pos_labels, train_size, val_size):
    train_indices, val_indices = random_split(train_size[1], val_size[1])
    train_P = get_image_dataset(dataset_name, pos_labels, num_labeled=train_size[0], indices=train_indices, train=True)
    train_U = get_image_dataset(dataset_name, pos_labels, indices=train_indices, train=True)
    val_P = get_image_dataset(dataset_name, pos_labels, num_labeled=val_size[0], indices=val_indices, train=True)
    val_U = get_image_dataset(dataset_name, pos_labels, indices=val_indices, train=True)
    return train_P, train_U, val_P, val_U


# load dataset for test
def load_testset(dataset_name, pos_labels, prior=None):
    testset = get_image_dataset(dataset_name, pos_labels, prior=prior, train=False)
    return testset


def generate_trainset(dataset_name, train_size, val_size, prior=0.5):
    train_P = get_synthetic_dataset(dataset_name, train_size[0], prior=1, labeled=True)
    train_U = get_synthetic_dataset(dataset_name, train_size[1], prior=prior, labeled=False)
    val_P = get_synthetic_dataset(dataset_name, val_size[0], prior=1, labeled=True)
    val_U = get_synthetic_dataset(dataset_name, val_size[1], prior=prior, labeled=True)
    return train_P, train_U, val_P, val_U


def generate_testset(dataset_name, test_size, prior=0.5):
    testset = get_synthetic_dataset(dataset_name, test_size, prior=prior, labeled=True)
    return testset


# training procedure
def train(device, method, dataset_name, datasets, loss_name, alpha, max_epochs, batch_size, stepsize, mixup_lam=None):
    # data setup ----------------
    trainset_P, trainset_U, valset_P, valset_U = datasets
    batch_num = len(trainset_U) // batch_size
    trainloader_P = torch.utils.data.DataLoader(trainset_P, batch_size=len(trainset_P)//batch_num, shuffle=True, drop_last=True, num_workers=2)
    trainloader_U = torch.utils.data.DataLoader(trainset_U, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    valloader_P = torch.utils.data.DataLoader(valset_P, batch_size=len(valset_P)//batch_num, shuffle=False, drop_last=False, num_workers=1)
    valloader_U = torch.utils.data.DataLoader(valset_U, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)

    # model setup ----------------
    activate_output = False if method in UNBIASED else True

    if dataset_name in SYNTHETIC:
        means = trainset_U[:][0].to(device)
        model = select_model(dataset_name)(means=means, activate_output=activate_output).to(device)
    else:
         model = select_model(dataset_name)(activate_output=activate_output).to(device)

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
    betas = (0.5, 0.999) if dataset_name in SYNTHETIC else (0.9, 0.999)
    weight_decay = 0.1 if dataset_name in SYNTHETIC else 0.005
    optimizer = torch.optim.Adam(params=model.parameters(), lr=stepsize, weight_decay=weight_decay, betas=betas)
    results = Results(["train_loss", "validation_loss"])

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
            if mixup_lam is not None:
                perm = np.random.permutation(np.arange(num_U))[:num_P]
                beta = np.random.beta(0.3, 0.3)
                x_mix = (beta * x_p + (1 - beta) * x_u[perm]).to(device)
                t_mix = (beta * y_p + (1 - beta) * y_u[perm]).to(device)
                y_mix = model(x_mix).view(-1)
                loss += mixup_lam * torch.square(y_mix - t_mix).mean()
            loss.backward()
            optimizer.step()
            train_loss.append(criterion.value())
        results.append("train_loss", np.array(train_loss).mean())

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
            results.append("validation_loss", np.array(validation_loss).mean())

        if ep % 20 == 19:
            stepsize /= 2
            optimizer = torch.optim.Adam(params=model.parameters(), lr=stepsize, weight_decay=weight_decay, betas=betas)

        print("Epoch {}, Train loss : {:.4f}, Val loss : {:.4f}".format(ep, results.get("train_loss"), results.get("validation_loss")))

    if method in UNBIASED:
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
    
    return model, train_prior, preds_P, results


# test procedure
def test(device, model, testset, train_prior, preds_P, batch_size=500):
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


def find_boundary(device, model, thresh, min_max):
    x = np.arange(round(min_max[0], 3), round(min_max[1], 3), 0.01)
    x_tensor = torch.from_numpy(x).float().reshape(len(x), 1).to(device)
    y = to_ndarray(model(x_tensor))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(x, y)
    fig.savefig("results/prediction.png")
    for i, p in enumerate(y):
        if i > 0 and y[i - 1] - thresh < 0 and y[i] - thresh >= 0:
            return x[i]
    return x[0] if x[0] - thresh > 0 else x[-1]


def run(device_num, method, dataset_name, loss_name, alpha, pos_labels, priors, train_size, validation_size, max_epochs, batch_size, stepsize, mixup_lam, path, synthetic_prior, seed, id):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    print("Device : {}".format(device))
    num_test = len(priors)
    home_directory = create_directory("{}/{}/{}-{}-{}".format(path, dataset_name, method, dataset_name, loss_name))

    # loading training dataset
    if dataset_name in SYNTHETIC:
        train_P, train_U, val_P, val_U = generate_trainset(dataset_name, train_size, validation_size, prior=synthetic_prior)
    else:
        train_P, train_U, val_P, val_U = load_trainset(dataset_name, pos_labels, train_size, validation_size)

    # estimate the class prior for uPU/nnPU
    if alpha is None:
        trans = Tensor_to_1darray()
        pos = np.concatenate([np.array([trans(x) for x, t in train_P]), np.array([trans(x) for x, t in val_P])])[:2000]
        neg = np.concatenate([np.array([trans(x) for x, t in train_U]), np.array([trans(x) for x, t in val_U])])[:2000]
        alpha = KM2_estimate(pos, neg) if method in UNBIASED else 0
        print("KM2 estimate : {:.4f}".format(alpha))
    
    # training
    print("Training started.")
    model, train_prior, preds_P, history = train(
        device=device,
        method=method,
        dataset_name=dataset_name,
        datasets=(train_P, train_U, val_P, val_U),
        loss_name=loss_name,
        alpha=alpha,
        max_epochs=max_epochs,
        batch_size=batch_size,
        stepsize=stepsize,
        mixup_lam=mixup_lam
    )
    # save training results
    history_path = create_directory(home_directory + "/train/history_{}".format(id))
    history.saveall(history_path)
    history.plotall(history_path)
    train_loss = history.get("train_loss")
    validation_loss = history.get("validation_loss")
    save_model(model, history_path)

    # output summary
    with open(home_directory + "/log_{}.txt".format(id), mode='a', encoding="utf-8") as f:
        print("Method : {}\nDataset : {}\nLoss function : {}\nAlpha : {}\n".format(method, dataset_name, loss_name, alpha), file=f)
        print("--Training result--\n", file=f)
        print("Train loss : {:.9f}".format(train_loss), file=f)
        print("Validation loss : {:.9f}\n".format(validation_loss), file=f)
        print("--Test result---\n", file=f)
    
    # loading test dataset
    if dataset_name in SYNTHETIC:
        testsets = [generate_testset(dataset_name, test_size=500, prior=priors[test_idx]) for test_idx in range(num_test)]
    else:
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
        if method in UNBIASED:
            pri = alpha
        if dataset_name in SYNTHETIC:
            bnd = find_boundary(device, model, ths, testsets[test_idx].min_max())
    
        # output results
        test_path = create_directory(home_directory + "/test-{}".format(test_idx + 1))
        with open(test_path + "/accuracy.txt", mode='a', encoding="utf-8") as f:
            print(acc, file=f)
        with open(test_path + "/auc.txt", mode='a', encoding="utf-8") as f:
            print(auc, file=f)
        with open(test_path + "/prior.txt", mode='a', encoding="utf-8") as f:
            print(pri, file=f)
        if method not in UNBIASED:
            with open(test_path + "/thresh.txt", mode='a', encoding="utf-8") as f:
                print(ths, file=f)
        if dataset_name in SYNTHETIC:
            with open(test_path + "/boundary.txt", mode='a', encoding="utf-8") as f:
                print(bnd, file=f)

        # output summary of test results
        with open(home_directory + "/log_{}.txt".format(id), mode='a', encoding="utf-8") as f:
            print("Test {} : prior = {}".format(test_idx + 1, priors[test_idx]), file=f)
            print("Accuracy : {:.4f}".format(acc), file=f)
            print("AUC : {:.4f}".format(auc), file=f)
            print("Prior : {:.6f}".format(pri), file=f)
            print("Thresh : {:.6f}".format(ths), file=f)
            if dataset_name in SYNTHETIC:
                print("Boundary : {:.6f}".format(bnd), file=f)
            print("", file=f)

    # output summary of parameters
    with open(home_directory + "/log_{}.txt".format(id), mode='a', encoding="utf-8") as f:
        print("--Parameters--", file=f)
        print("train_size = {}".format(train_size), file=f)
        print("validation_size = {}".format(validation_size), file=f)
        print("max_epochs = {}".format(max_epochs), file=f)
        print("batch_size = {}".format(batch_size), file=f)
        print("stepsize = {}".format(stepsize), file=f)
        print("random seed = {}".format(seed), file=f)

    return validation_loss