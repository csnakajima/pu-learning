import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


class Accuracy(object):
    def __call__(self, y, t):
        return np.mean(y * t >= 0)


class AUROC(object):
    def __call__(self, y, t):
        auc = roc_auc_score(t, y)
        self.fpr, self.tpr, self.thresh = roc_curve(t, y, drop_intermediate=False)
        return auc

    def roc_curve(self):
        return self.fpr, self.tpr, self.thresh


class PURiskEstimator(object):
    def __init__(self, prior, loss):
        self.prior = prior
        self.loss = loss
        self.L = None

    def __call__(self, y_p, y_u):
        E_p = torch.mean(self.loss(y_p) - self.loss(-y_p))
        E_u = torch.mean(self.loss(-y_u))
        self.L = self.prior * E_p + E_u
        return self.L

    def value(self):
        return self.L.item()


class NonNegativeRiskEstimator(PURiskEstimator):
    def __init__(self, prior, loss, thresh=0, weight=1):
        super().__init__(prior, loss)
        self.thresh = thresh
        self.weight = weight
    
    def __call__(self, y_p, y_u):
        E_pp = torch.mean(self.loss(y_p))
        E_pn = torch.mean(self.loss(-y_p))
        E_u = torch.mean(self.loss(-y_u))
        self.L = self.prior * E_pp + max(0, E_u - self.prior * E_pn)
        return self.L if E_u - self.prior * E_pn >= self.thresh else self.weight * (self.prior * E_pn - E_u)


class BregmanDivergence(object):
    def __init__(self, f_df):
        self.f = f_df[0]
        self.df = f_df[1]
        self.L = None
    
    def __call__(self, y_p, y_u):
        E_p = torch.mean(-self.df(y_p))
        E_u = torch.mean(y_u * self.df(y_u) - self.f(y_u))
        self.L = E_p + E_u
        return self.L

    def value(self):
        return self.L.item()


class NonNegativeBregmanDivergence(BregmanDivergence):
    def __init__(self, prior, f_df, thresh=0, weight=1):
        super().__init__(f_df)
        self.prior = prior
        self.thresh = thresh
        self.weight = weight
        self.f_res = lambda x: self.df(x) - self.prior * (x * self.df(x) - self.f(x))
        self.loss = lambda x: x * self.df(x) - self.f(x) + 1/2

    def __call__(self, y_p, y_u):
        E_pp = torch.mean(-self.f_res(y_p))
        E_pn = torch.mean(self.loss(y_p))
        E_u = torch.mean(self.loss(y_u))
        self.L = E_pp + max(0, E_u - self.prior * E_pn)
        return self.L if E_u - self.prior * E_pn >= self.thresh else self.weight * (self.prior * E_pn - E_u)


def select_loss(loss_name):
    losses = {  
        "zero-one": lambda x: (torch.sign(-x) + 1) / 2,
        "sigmoid": lambda x: torch.sigmoid(-x),
        "logistic": lambda x: F.softplus(-x),
        "squared": lambda x: torch.square(x - 1) / 2,
        "savage": lambda x: 4 / torch.square(1 + torch.exp(x)),
        "LSIF": (lambda x: torch.square(x - 1) / 2, lambda x: x - 1)
    }
    return losses[loss_name]