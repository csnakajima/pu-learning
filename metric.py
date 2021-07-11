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


class AsymmetricNonNegativeRiskEstimator(NonNegativeRiskEstimator):
    def __init__(self, train_prior, test_prior, loss, thresh=0, weight=1):
        super().__init__(train_prior, loss, thresh, weight)
        self.train_prior = train_prior
        self.test_prior = test_prior

    def __call__(self, y_p, y_u):
        E_pp = torch.mean(self.loss(y_p))
        E_pn = torch.mean(self.loss(-y_p))
        E_u = torch.mean(self.loss(-y_u))
        self.L = self.test_prior * E_pp + (1 - self.test_prior) / (1 - self.train_prior) * max(0, E_u - self.train_prior * E_pn)
        return self.L if E_u - self.train_prior * E_pn >= self.thresh else self.weight * (self.train_prior * E_pn - E_u)


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
    def __init__(self, alpha, f_df, thresh=0, weight=1):
        super().__init__(f_df)
        self.alpha = alpha
        self.thresh = thresh
        self.weight = weight
        self.f_dual = lambda x: x * self.df(x) -self.f(x)
        self.f_nn = lambda x: self.f_dual(x) - self.f_dual(0 * x)

    def __call__(self, y_p, y_u):
        E_pp = torch.mean(-self.df(y_p) + self.alpha * self.f_nn(y_p))
        E_pn = torch.mean(self.f_nn(y_p))
        E_u = torch.mean(self.f_nn(y_u))
        self.L = E_pp + max(0, E_u - self.alpha * E_pn) + self.f_dual(0 * E_u)
        return self.L if E_u - self.alpha * E_pn >= self.thresh else self.weight * (self.alpha * E_pn - E_u)


def choose_loss(loss_name):
    losses = {  
        "zero-one": lambda x: (torch.sign(-x) + 1) / 2,
        "sigmoid": lambda x: torch.sigmoid(-x),
        "logistic": lambda x: F.softplus(-x),
        "squared": lambda x: torch.square(x - 1) / 2,
        "savage": lambda x: 4 / torch.square(1 + torch.exp(x)),
        "LSIF": (lambda x: torch.square(x - 1) / 2, lambda x: x - 1)
    }
    return losses[loss_name]

