import numpy as np
import torch

from metric import Accuracy, AUROC
from save import Results
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-16

def to_ndarray(tensor):
    return tensor.to('cpu').detach().numpy().copy()


def PUsequence(num_P, num_U):
    return np.array([1] * num_P + [-1] * num_U)


# y, t : ndarray
def priorestimator(y, t):
    num_P = np.count_nonzero(t == 1)
    delta = 1 / num_P
    eps = np.sqrt(4 * np.log(np.exp(1) * num_P / 2) / num_P) + np.sqrt(np.log(2 / delta) / (2 * num_P))
    fpr, tpr, thresh = roc_curve(t, y, drop_intermediate=False)
    idx = tpr > eps
    ratio = fpr[idx] / tpr[idx]
    ratio = ratio[~(np.isnan(ratio) + np.isinf(ratio))]
    assert np.size(ratio) > 0
    return np.min(ratio)


def ERM(model, optimizer, trainloader_P, trainloader_U, valloader_P, valloader_U, testloaders, criterion, criterion_val, max_epochs, device, given_thresholds=None):
    train_result = Results(["train_loss", "validation_loss"])
    test_results = [Results(["accuracy", "auc", "prior", "thresh"]) for i in range(len(testloaders))] if given_thresholds is None else [Results(["accuracy", "auc"]) for i in range(len(testloaders))]
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
        train_result.append("train_loss", np.array(train_loss).mean())

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
            train_result.append("validation_loss", np.array(validation_loss).mean())

        # test step
        with torch.no_grad():
            for i, testloader in enumerate(testloaders):
                if given_thresholds is None:
                    train_prior, preds_P = estimate_train_prior(model, valloader_P, valloader_U, device)
                    test_prior = estimate_test_prior(model, testloader, preds_P, device)
                    thresh = train_prior * (1 - test_prior) / (train_prior * ((1 - train_prior) * test_prior + train_prior * (1 - test_prior)) + EPS)
                    test_results[i].append("prior", test_prior)
                    test_results[i].append("thresh", thresh)
                else:
                    thresh = given_thresholds[i]
                acc, auc = prediction(model, testloader, device, thresh)
                test_results[i].append("accuracy", acc)
                test_results[i].append("auc", auc)

        if ep % 20 == 19:
            optimizer.param_groups[0]['lr'] /= 2

        print("Epoch {}, Train loss : {:.4f}, Val loss : {:.4f}".format(ep, train_result.get("train_loss"), train_result.get("validation_loss")))
    
    return model, train_result, test_results


def estimate_train_prior(model, valloader_P, valloader_U, device):
    model.eval()
    with torch.no_grad():
        preds_P, preds_U = [], []
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
        train_prior = priorestimator(np.concatenate([preds_P, preds_U]), PUsequence(len(preds_P), len(preds_U)))
    return train_prior, preds_P


def estimate_test_prior(model, testloader, preds_P, device):
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
        test_prior = priorestimator(np.concatenate([preds_P, preds_U]), PUsequence(len(preds_P), len(preds_U)))
    return test_prior


def prediction(model, testloader, device, thresh=0):
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
        acc = accuracy(preds_U - thresh, targets)
        auc = auroc(preds_U, targets)
    return acc, auc


def find_boundary(model, min_max, device, thresh=0):
    x = np.arange(round(min_max[0], 3), round(min_max[1], 3), 0.01)
    x_tensor = torch.from_numpy(x).float().reshape(len(x), 1).to(device)
    y = to_ndarray(model(x_tensor))
    for i, p in enumerate(y):
        if i > 0 and y[i - 1] - thresh < 0 and y[i] - thresh >= 0:
            return x[i]
    return x[0] if x[0] - thresh > 0 else x[-1]

