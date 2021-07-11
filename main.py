from operator import pos
import sys
import argparse
import numpy as np
import torch

from dataset import ImageDataset

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of unbiased/non-negative/density-ratio PU learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--method", "-m", type=str, default="DRPU", choices=["uPU", "nnPU", "PUa", "DRPU"],
                        help="Learning algorithm")
    parser.add_argument("--dataset", "-d", type=str, default="mnist", choices=["gauss", "gauss_mix", "mnist", "fmnist", "kmnist", "cifar"],
                        help="Dataset name")
    parser.add_argument("--num_positive", "-n", type=int, default=2500,
                        help="Number of positively labeled data for training dataset")
    parser.add_argument("--loss", "-l", type=str, default="LSIF", choices=["sigmoid", "logistic", "savage", "LSIF"],
                        help="Loss function name")
    parser.add_argument("--alpha", "-a", type=float, default=None,
                        help="Parameter for risk estimator")
    parser.add_argument("--preset", "-p", action="store_true",
                        help="Use preset of parameter settings")
    parser.add_argument("--max_epochs",  type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--oracle_prior", action="store_true",
                        help="Use oracle class-priors (for benchmark data)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--res_dir", type=str, default="results",
                        help="Directory to output results")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory of datasets")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--id", "-i", type=int, default=None,
                        help="Job ID")
    args = parser.parse_args(arguments)
    if args.method in ["DRPU"]:
        if args.loss not in ["LSIF"]:
            args.loss = "LSIF"
    else:
        if args.loss in ["LSIF"]:
            args.loss = "sigmoid"
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args


def main(args):
    args = process_args(args)

    if args.dataset in ["gauss", "gauss_mix"]:
        true_test_priors = [0.4, 0.6]
        train_size = (200, 1000)
        val_size = (100, 500)
        test_size = 500
        max_epochs = 200
        batch_size = 100
        lr = 2e-5
        alpha = None
        loss_name = args.loss
        true_train_prior = 0.4 if args.dataset == "gauss" else 0.6

        if not args.preset:
            train_size = (args.num_positive, train_size[1])
            max_epochs = args.max_epochs
            batch_size = args.batch_size
            lr = args.lr
            alpha = args.alpha
            loss_name = "LSIF" if args.method == "DRPU" else "logistic"
        if args.method == "uPU":
            from run_synthetic import uPU as run
        else:
            from run_synthetic import DRPU as run

        run(
            dataset_name=args.dataset,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            alpha=alpha,
            loss_name=loss_name,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            true_train_prior=true_train_prior,
            true_test_priors=true_test_priors,
            device_num=args.gpu,
            res_dir=args.res_dir,
            seed=args.seed,
            id=args.id
        )
    
    else:
        true_test_priors = [0.2, 0.4, 0.6, 0.8]
        loss_name = "LSIF" if args.method == "DRPU" else "sigmoid"
        if args.dataset == "mnist":
            ImageDataset.pos_labels = [0, 2, 4, 6, 8]
            true_train_prior = 0.5 if args.oracle_prior else None
            train_size = (2500, 50000)
            val_size = (500, 10000)
            max_epochs = 50
            batch_size = 500
            lr = 2e-5
            alpha = 0.475 if args.method == "DRPU" else None
        elif args.dataset == "fmnist":
            ImageDataset.pos_labels = [2, 3, 4, 5, 6, 9]
            true_train_prior = 0.6 if args.oracle_prior else None
            train_size = (2500, 50000)
            val_size = (500, 10000)
            max_epochs = 100
            batch_size = 500
            lr = 2e-5
            alpha = 0.6 if args.method == "DRPU" else None
        elif args.dataset == "kmnist":
            ImageDataset.pos_labels = [0, 1, 8, 9]
            true_train_prior = 0.4 if args.oracle_prior else None
            train_size = (2500, 50000)
            val_size = (500, 10000)
            max_epochs = 100
            batch_size = 500
            lr = 2e-5
            alpha = 0.375 if args.method == "DRPU" else None
        else:
            ImageDataset.pos_labels = [0, 1, 8, 9]
            true_train_prior = 0.4 if args.oracle_prior else None
            train_size = (2500, 45000)
            val_size = (500, 5000)
            max_epochs = 100
            batch_size = 500
            lr = 1e-5
            alpha = 0.425 if args.method == "DRPU" else None
        
        if not args.preset:
            train_size = (args.num_positive, train_size[1])
            max_epochs = args.max_epochs
            batch_size = args.batch_size
            lr = args.lr
            alpha = args.alpha
            loss_name = args.loss

        if args.method == "uPU":
            from run_benchmark import uPU as run
        elif args.method == "nnPU":
            from run_benchmark import nnPU as run
        elif args.method == "PUa":
            from run_benchmark import PUa as run
        else:
            from run_benchmark import DRPU as run

        run(
            dataset_name=args.dataset,
            train_size=train_size,
            val_size=val_size,
            alpha=alpha,
            loss_name=loss_name,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            true_train_prior=true_train_prior,
            true_test_priors=true_test_priors,
            device_num=args.gpu,
            res_dir=args.res_dir,
            data_dir=args.data_dir,
            seed=args.seed,
            id=args.id
        )


if __name__ == '__main__':
    main(sys.argv[1:])

