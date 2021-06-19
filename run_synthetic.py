import numpy as np
from numpy.lib.function_base import append
import torch
import os

from dataset import get_synthetic_positive, get_synthetic_test, get_synthetic_unlabeled, Tensor_to_1darray
from model import choose_model
from metric import *
from save import *
from algorithm import *
from Kernel_MPE import KM2_estimate


def load_trainset(dataset_name, train_size, val_size, batch_size, prior):
    trainsize_P, trainsize_U = train_size
    valsize_P, valsize_U = val_size
    trainset_P = get_synthetic_positive(dataset_name, trainsize_P)
    trainset_U = get_synthetic_unlabeled(dataset_name, trainsize_U, prior)
    valset_P = get_synthetic_positive(dataset_name, valsize_P)
    valset_U = get_synthetic_unlabeled(dataset_name, valsize_U, prior)
    batch_num = len(trainset_U) // batch_size
    trainloader_P = torch.utils.data.DataLoader(trainset_P, batch_size=len(trainset_P)//batch_num, shuffle=True, drop_last=True, num_workers=2)
    trainloader_U = torch.utils.data.DataLoader(trainset_U, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    valloader_P = torch.utils.data.DataLoader(valset_P, batch_size=len(valset_P)//batch_num, shuffle=False, drop_last=False, num_workers=1)
    valloader_U = torch.utils.data.DataLoader(valset_U, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return trainloader_P, trainloader_U, valloader_P, valloader_U, trainset_P, trainset_U, valset_P, valset_U


def load_testset(dataset_name, test_size, batch_size, prior):
    testset = get_synthetic_test(dataset_name, test_size, prior)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return testloader, testset




def uPU(dataset_name, train_size, val_size, test_size, alpha, loss_name, max_epochs, batch_size, lr, true_train_prior, true_test_priors, device_num, res_dir, seed, id):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    res_dir = getdirs(os.path.join(os.getcwd(), "{}/{}/{}".format(res_dir, "uPU", dataset_name)))
    
    trainloader_P, trainloader_U, valloader_P, valloader_U, trainset_P, trainset_U, _, _ = load_trainset(dataset_name, train_size, val_size, batch_size, true_train_prior)
    
    if alpha is None:
        trans = Tensor_to_1darray()
        pos = np.array([trans(x) for x, t in trainset_P])[:2000]
        unl = np.array([trans(x) for x, t in trainset_U])[:2000]
        train_prior = KM2_estimate(pos, unl)
    else:
        train_prior = alpha

    means = trainset_U[:][0].to(device)
    model = choose_model(dataset_name)(means).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.1, betas=(0.5, 0.999))
    criterion = PURiskEstimator(train_prior, choose_loss(loss_name))
    criterion_val = PURiskEstimator(train_prior, loss=choose_loss("zero-one"))

    model, history = ERM(
        model=model,
        optimizer=optimizer,
        trainloader_P=trainloader_P,
        trainloader_U=trainloader_U,
        valloader_P=valloader_P,
        valloader_U=valloader_U,
        criterion=criterion,
        criterion_val=criterion_val,
        max_epochs=max_epochs,
        device=device
    )

    save_train_history(getdirs(os.path.join(res_dir, "train/history_{}".format(id))), model, history)
    output_train_results(os.path.join(res_dir, "log_{}.txt".format(id)), history, train_prior)

    for i, true_prior in enumerate(true_test_priors):
        testloader, testset = load_testset(dataset_name, test_size, batch_size, true_prior)
        acc, auc = prediction(model, testloader, device)
        boundary = find_boundary(model, min_max=testset.min_max(), device=device)
        output_test_results(os.path.join(res_dir, "log_{}.txt".format(id)), i, true_prior, acc, auc, boundary)
        append_test_results(getdirs(os.path.join(res_dir, "test-{}".format(i))), acc, auc, boundary)

    output_config(os.path.join(res_dir, "log_{}.txt".format(id)), train_size, val_size, max_epochs, batch_size, lr, alpha, seed)




def DRPU(dataset_name, train_size, val_size, test_size, alpha, loss_name, max_epochs, batch_size, lr, true_train_prior, true_test_priors, device_num, res_dir, seed, id):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    res_dir = getdirs(os.path.join(os.getcwd(), "{}/{}/{}".format(res_dir, "DRPU", dataset_name)))
    
    trainloader_P, trainloader_U, valloader_P, valloader_U, _, trainset_U, _, _ = load_trainset(dataset_name, train_size, val_size, batch_size, true_train_prior)

    if alpha is None:
        alpha = 0
    means = trainset_U[:][0].to(device)
    model = choose_model(dataset_name)(means, activate_output=True).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.1, betas=(0.5, 0.999))
    criterion = BregmanDivergence(choose_loss(loss_name))
    criterion_val = BregmanDivergence(choose_loss(loss_name))

    model, history = ERM(
        model=model,
        optimizer=optimizer,
        trainloader_P=trainloader_P,
        trainloader_U=trainloader_U,
        valloader_P=valloader_P,
        valloader_U=valloader_U,
        criterion=criterion,
        criterion_val=criterion_val,
        max_epochs=max_epochs,
        device=device
    )

    train_prior, preds_P = estimate_train_prior(model, valloader_P, valloader_U, device)

    save_train_history(getdirs(os.path.join(res_dir, "train/history_{}".format(id))), model, history)
    output_train_results(os.path.join(res_dir, "log_{}.txt".format(id)), history, train_prior)

    for i, true_prior in enumerate(true_test_priors):
        testloader, testset = load_testset(dataset_name, test_size, batch_size, true_prior)
        test_prior = estimate_test_prior(model, testloader, preds_P, device)
        thresh = train_prior * (1 - test_prior) / ((1 - train_prior) * test_prior + train_prior * (1 - test_prior))
        thresh /= train_prior
        acc, auc = prediction(model, testloader, device, thresh)
        boundary = find_boundary(model, min_max=testset.min_max(), device=device, thresh=thresh)
        output_test_results(os.path.join(res_dir, "log_{}.txt".format(id)), i, true_prior, acc, auc, test_prior, thresh, boundary)
        append_test_results(getdirs(os.path.join(res_dir, "test-{}".format(i))), acc, auc, test_prior, thresh, boundary)

    output_config(os.path.join(res_dir, "log_{}.txt".format(id)), train_size, val_size, max_epochs, batch_size, lr, alpha, seed)
