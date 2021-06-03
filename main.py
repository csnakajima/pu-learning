import sys
import argparse
import numpy as np
import torch

from train import run, SYNTHETIC, UNBIASED

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of unbiased/non-negative/density-ratio PU learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gpu", "-g", type=int, default=0,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--method", "-m", type=str, default="nnDRPU", choices=["uPU", "nnPU", "DRPU"],
                        help="Learning algorithm")
    parser.add_argument("--dataset", "-d", type=str, default="mnist", choices=["gauss", "gauss_mix", "mnist", "fmnist", "kmnist", "cifar"],
                        help="Dataset name")
    parser.add_argument("--loss", "-l", type=str, default="LSIF", choices=["sigmoid", "logistic", "savage", "LSIF"],
                        help="Loss function")
    parser.add_argument("--alpha", "-a", type=float, default=None,
                        help="Parameter for the risk estimator")
    parser.add_argument("--path", "-p", type=str, default="results",
                        help="Directory to output the results")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--id", "-i", type=int, default=None,
                        help="Job ID")
    args = parser.parse_args(arguments)
    if args.method in UNBIASED:
        assert args.loss not in ["LSIF"]
    else:
        assert args.loss in ["LSIF"]
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args


def main(args):
    args = process_args(args)
    device_num = args.gpu
    method = args.method
    dataset_name = args.dataset
    loss_name = args.loss
    alpha = args.alpha
    mixup_lam = None
    path = args.path
    synthetic_prior = None
    seed = args.seed
    id = args.id
    priors = [0.2, 0.4, 0.6, 0.8]
    # presets
    if dataset_name in SYNTHETIC:
        pos_labels = []
        priors = [0.4, 0.6]
        train_size = (200, 1000)
        validation_size = (100, 500)
        max_epochs = 200
        batch_size = 100
        stepsize = 2e-5
        synthetic_prior = 0.4 if dataset_name == "gauss" else 0.6
    elif dataset_name == "mnist":
        pos_labels = [0, 2, 4, 6, 8]
        train_size = (2500, 50000)
        validation_size = (500, 10000)
        max_epochs = 50
        batch_size = 500
        stepsize = 2e-5
    elif dataset_name == "fmnist":
        pos_labels = [2, 3, 4, 5, 6, 9]
        train_size = (2500, 50000)
        validation_size = (500, 10000)
        max_epochs = 100
        batch_size = 500
        stepsize = 2e-5
    elif dataset_name == "kmnist":
        pos_labels = [0, 1, 8, 9]
        train_size = (2500, 50000)
        validation_size = (500, 10000)
        max_epochs = 100
        batch_size = 500
        stepsize = 2e-5
    elif dataset_name == "cifar":
        pos_labels = [0, 1, 8, 9]
        train_size = (2500, 45000)
        validation_size = (500, 5000)
        max_epochs = 100
        batch_size = 500
        stepsize = 1e-5

    run(device_num=device_num,
        method=method,
        dataset_name=dataset_name,
        loss_name=loss_name,
        alpha=alpha,
        pos_labels=pos_labels,
        priors=priors,
        train_size=train_size,
        validation_size=validation_size,
        max_epochs=max_epochs,
        batch_size=batch_size,
        stepsize=stepsize,
        path=path,
        synthetic_prior=synthetic_prior,
        mixup_lam=mixup_lam,
        seed=seed,
        id=id
    )


if __name__ == '__main__':
    main(sys.argv[1:])

