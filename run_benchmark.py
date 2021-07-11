import numpy as np
from numpy.lib.function_base import append
import torch
import os

from dataset import get_image_positive, get_image_test, get_image_unlabeled, Tensor_to_1darray
from model import choose_model
from metric import *
from save import *
from algorithm import *
from modules.Kernel_MPE import KM2_estimate


def random_split(train_size, val_size):
    indices = np.arange(train_size + val_size)
    np.random.shuffle(indices)
    return indices[:train_size], indices[-val_size:]


def load_trainset(dataset_name, train_size, val_size, batch_size, data_dir="dataset"):
    trainsize_P, trainsize_U = train_size
    valsize_P, valsize_U = val_size
    train_indices, val_indices = random_split(trainsize_U, valsize_U)
    trainset_P = get_image_positive(dataset_name, trainsize_P, train_indices, root=data_dir)
    trainset_U = get_image_unlabeled(dataset_name, train_indices, root=data_dir)
    valset_P = get_image_positive(dataset_name, valsize_P, val_indices, root=data_dir)
    valset_U = get_image_unlabeled(dataset_name, val_indices, root=data_dir)
    batch_num = len(trainset_U) // batch_size
    trainloader_P = torch.utils.data.DataLoader(trainset_P, batch_size=len(trainset_P)//batch_num, shuffle=True, drop_last=True, num_workers=2)
    trainloader_U = torch.utils.data.DataLoader(trainset_U, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    valloader_P = torch.utils.data.DataLoader(valset_P, batch_size=len(valset_P)//batch_num, shuffle=False, drop_last=False, num_workers=1)
    valloader_U = torch.utils.data.DataLoader(valset_U, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return trainloader_P, trainloader_U, valloader_P, valloader_U, trainset_P, trainset_U, valset_P, valset_U


def load_testset(dataset_name, batch_size, prior=None, data_dir="dataset"):
    testset = get_image_test(dataset_name, prior, root=data_dir)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)
    return testloader, testset




def uPU(dataset_name, train_size, val_size, alpha, loss_name, max_epochs, batch_size, lr, true_train_prior, true_test_priors, device_num, res_dir, data_dir, seed, id):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    res_dir = getdirs(os.path.join(os.getcwd(), res_dir, "uPU", dataset_name))
    data_dir = getdirs(os.path.join(os.getcwd(), data_dir))
    
    trainloader_P, trainloader_U, valloader_P, valloader_U, trainset_P, trainset_U, _, _ = load_trainset(dataset_name, train_size, val_size, batch_size, data_dir)
    
    if true_train_prior is not None:
        train_prior = true_train_prior
    elif alpha is None:
        trans = Tensor_to_1darray()
        pos = np.array([trans(x) for x, t in trainset_P])[:2000]
        unl = np.array([trans(x) for x, t in trainset_U])[:2000]
        train_prior = KM2_estimate(pos, unl)
    else:
        train_prior = alpha

    testloaders = [load_testset(dataset_name, batch_size, true_prior, data_dir)[0] for true_prior in true_test_priors]

    model = choose_model(dataset_name)().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.005, betas=(0.9, 0.999))
    criterion = PURiskEstimator(train_prior, choose_loss(loss_name))
    criterion_val = PURiskEstimator(train_prior, loss=choose_loss("zero-one"))

    model, train_result, test_results = ERM(
        model=model,
        optimizer=optimizer,
        trainloader_P=trainloader_P,
        trainloader_U=trainloader_U,
        valloader_P=valloader_P,
        valloader_U=valloader_U,
        testloaders=testloaders,
        criterion=criterion,
        criterion_val=criterion_val,
        max_epochs=max_epochs,
        device=device,
        given_thresholds=[0]*len(true_test_priors)
    )

    save_train_history(getdirs(os.path.join(res_dir, "train", "history_{}".format(id))), model, train_result)
    output_train_results(os.path.join(res_dir, "log_{}.txt".format(id)), train_result, train_prior)
    save_test_history(getdirs(os.path.join(res_dir, "train", "history_{}".format(id))), test_results)

    for i, (true_prior, result) in enumerate(zip(true_test_priors, test_results)):
        acc = result.get("accuracy")
        auc = result.get("auc")
        output_test_results(os.path.join(res_dir, "log_{}.txt".format(id)), i, true_prior, acc, auc)
        append_test_results(getdirs(os.path.join(res_dir, "test-{}".format(i))), acc, auc)

    output_config(os.path.join(res_dir, "log_{}.txt".format(id)), train_size, val_size, max_epochs, batch_size, lr, alpha, seed)




def nnPU(dataset_name, train_size, val_size, alpha, loss_name, max_epochs, batch_size, lr, true_train_prior, true_test_priors, device_num, res_dir, data_dir, seed, id):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    res_dir = getdirs(os.path.join(os.getcwd(), res_dir, "nnPU", dataset_name))
    data_dir = getdirs(os.path.join(os.getcwd(), data_dir))
    
    trainloader_P, trainloader_U, valloader_P, valloader_U, trainset_P, trainset_U, _, _ = load_trainset(dataset_name, train_size, val_size, batch_size, data_dir)
    
    if true_train_prior is not None:
        train_prior = true_train_prior
    elif alpha is None:
        trans = Tensor_to_1darray()
        pos = np.array([trans(x) for x, t in trainset_P])[:2000]
        unl = np.array([trans(x) for x, t in trainset_U])[:2000]
        train_prior = KM2_estimate(pos, unl)
    else:
        train_prior = alpha

    testloaders = [load_testset(dataset_name, batch_size, true_prior, data_dir)[0] for true_prior in true_test_priors]

    model = choose_model(dataset_name)().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.005, betas=(0.9, 0.999))
    criterion = NonNegativeRiskEstimator(train_prior, choose_loss(loss_name))
    criterion_val = PURiskEstimator(train_prior, loss=choose_loss("zero-one"))

    model, train_result, test_results = ERM(
        model=model,
        optimizer=optimizer,
        trainloader_P=trainloader_P,
        trainloader_U=trainloader_U,
        valloader_P=valloader_P,
        valloader_U=valloader_U,
        testloaders=testloaders,
        criterion=criterion,
        criterion_val=criterion_val,
        max_epochs=max_epochs,
        device=device,
        given_thresholds=[0]*len(true_test_priors)
    )

    save_train_history(getdirs(os.path.join(res_dir, "train", "history_{}".format(id))), model, train_result)
    output_train_results(os.path.join(res_dir, "log_{}.txt".format(id)), train_result, train_prior)
    save_test_history(getdirs(os.path.join(res_dir, "train", "history_{}".format(id))), test_results)

    for i, (true_prior, result) in enumerate(zip(true_test_priors, test_results)):
        acc = result.get("accuracy")
        auc = result.get("auc")
        output_test_results(os.path.join(res_dir, "log_{}.txt".format(id)), i, true_prior, acc, auc)
        append_test_results(getdirs(os.path.join(res_dir, "test-{}".format(i))), acc, auc)

    output_config(os.path.join(res_dir, "log_{}.txt".format(id)), train_size, val_size, max_epochs, batch_size, lr, alpha, seed)




def PUa(dataset_name, train_size, val_size, alpha, loss_name, max_epochs, batch_size, lr, true_train_prior, true_test_priors, device_num, res_dir, data_dir, seed, id):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    res_dir = getdirs(os.path.join(os.getcwd(), res_dir, "PUa", dataset_name))
    data_dir = getdirs(os.path.join(os.getcwd(), data_dir))
    
    trainloader_P, trainloader_U, valloader_P, valloader_U, trainset_P, trainset_U, _, _ = load_trainset(dataset_name, train_size, val_size, batch_size, data_dir)
    
    if true_train_prior is not None:
        train_prior = true_train_prior
    elif alpha is None:
        trans = Tensor_to_1darray()
        pos = np.array([trans(x) for x, t in trainset_P])[:2000]
        unl = np.array([trans(x) for x, t in trainset_U])[:2000]
        train_prior = KM2_estimate(pos, unl)
    else:
        train_prior = alpha

    for i, true_prior in enumerate(true_test_priors):
        testloader, testset = load_testset(dataset_name, batch_size, true_prior, data_dir)
        trans = Tensor_to_1darray()
        unl = np.array([trans(x) for x, t in testset])[:2000]
        test_prior = KM2_estimate(pos, unl)
        model = choose_model(dataset_name)().to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.005, betas=(0.9, 0.999))
        criterion = AsymmetricNonNegativeRiskEstimator(train_prior, test_prior, choose_loss(loss_name))
        criterion_val = PURiskEstimator(test_prior, loss=choose_loss("zero-one"))

        model, train_result, test_results = ERM(
            model=model,
            optimizer=optimizer,
            trainloader_P=trainloader_P,
            trainloader_U=trainloader_U,
            valloader_P=valloader_P,
            valloader_U=valloader_U,
            testloaders=[testloader],
            criterion=criterion,
            criterion_val=criterion_val,
            max_epochs=max_epochs,
            device=device,
            given_thresholds=[0]*len(true_test_priors)
        )

        save_train_history(getdirs(os.path.join(res_dir, "train-{}".format(i), "history_{}".format(id))), model, train_result)
        if i == 0:
            output_train_results(os.path.join(res_dir, "log_{}.txt".format(id)), train_result, train_prior)
        test_results[0].saveall(getdirs(os.path.join(res_dir, "train-{}".format(i), "history_{}".format(id))), i)

        acc = test_results[0].get("accuracy")
        auc = test_results[0].get("auc")
        output_test_results(os.path.join(res_dir, "log_{}.txt".format(id)), i, true_prior, acc, auc, test_prior)
        append_test_results(getdirs(os.path.join(res_dir, "test-{}".format(i))), acc, auc, test_prior)

    output_config(os.path.join(res_dir, "log_{}.txt".format(id)), train_size, val_size, max_epochs, batch_size, lr, alpha, seed)




def DRPU(dataset_name, train_size, val_size, alpha, loss_name, max_epochs, batch_size, lr, true_train_prior, true_test_priors, device_num, res_dir, data_dir, seed, id):
    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    res_dir = getdirs(os.path.join(os.getcwd(), res_dir, "DRPU", dataset_name))
    data_dir = getdirs(os.path.join(os.getcwd(), data_dir))
    
    trainloader_P, trainloader_U, valloader_P, valloader_U, _, _, _, _ = load_trainset(dataset_name, train_size, val_size, batch_size, data_dir)

    if true_train_prior is not None:
        alpha = true_train_prior
    elif alpha is None:
        alpha = 0

    testloaders = [load_testset(dataset_name, batch_size, true_prior, data_dir)[0] for true_prior in true_test_priors]

    model = choose_model(dataset_name)(activate_output=True).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.005, betas=(0.9, 0.999))
    criterion = NonNegativeBregmanDivergence(alpha, choose_loss(loss_name))
    criterion_val = BregmanDivergence(choose_loss(loss_name))

    if true_train_prior is None:
        given_thresholds = None
    else:
        train_prior = true_train_prior
        given_thresholds = [train_prior * (1 - test_prior) / (train_prior * ((1 - train_prior) * test_prior + train_prior * (1 - test_prior)) + EPS)
                            for test_prior in true_test_priors]

    model, train_result, test_results = ERM(
        model=model,
        optimizer=optimizer,
        trainloader_P=trainloader_P,
        trainloader_U=trainloader_U,
        valloader_P=valloader_P,
        valloader_U=valloader_U,
        testloaders=testloaders,
        criterion=criterion,
        criterion_val=criterion_val,
        max_epochs=max_epochs,
        device=device,
        given_thresholds=given_thresholds
    )

    train_prior, preds_P = estimate_train_prior(model, valloader_P, valloader_U, device)

    save_train_history(getdirs(os.path.join(res_dir, "train", "history_{}".format(id))), model, train_result)
    save_test_history(getdirs(os.path.join(res_dir, "train", "history_{}".format(id))), test_results)
    output_train_results(os.path.join(res_dir, "log_{}.txt".format(id)), train_result, train_prior)

    for i, (true_prior, result) in enumerate(zip(true_test_priors, test_results)):
        acc = result.get("accuracy")
        auc = result.get("auc")
        prior = result.get("prior")
        thresh = result.get("thresh")
        output_test_results(os.path.join(res_dir, "log_{}.txt".format(id)), i, true_prior, acc, auc, prior, thresh)
        append_test_results(getdirs(os.path.join(res_dir, "test-{}".format(i))), acc, auc, prior, thresh)

    output_config(os.path.join(res_dir, "log_{}.txt".format(id)), train_size, val_size, max_epochs, batch_size, lr, alpha, seed)

