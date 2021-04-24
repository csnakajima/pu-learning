import sys
import argparse

from train import run


def process_args(arguments):
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of unbiased/non-negative/density-ratio PU learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gpu", "-g", type=int, default=0,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--trial", "-t", type=int, default=5,
                        help="Number of experiments")
    parser.add_argument("--method", "-m", type=str, default="DRPU", choices=["uPU", "nnPU", "DRPU"],
                        help="Learning algorithm")
    parser.add_argument("--dataset", "-d", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10"],
                        help="Dataset name")
    parser.add_argument("--loss", "-l", type=str, default="LSIF", choices=["sigmoid", "logistic", "savage", "LSIF"],
                        help="Loss function")
    parser.add_argument("--alpha", "-a", type=float, default=0.5,
                        help="Parameter for the risk estimator")
    parser.add_argument("--path", "-p", type=str, default="results",
                        help="Directory to output the results")
    args = parser.parse_args(arguments)
    if args.method in ["uPU", "nnPU"]:
        assert args.loss not in ["LSIF"]
    else:
        assert args.loss in ["LSIF"]
    return args


def main(args):
    args = process_args(args)
    device_num = args.gpu
    trial = args.trial
    method = args.method
    dataset_name = args.dataset
    loss_name = args.loss
    alpha = args.alpha
    path = args.path
    # presets
    if dataset_name == "mnist":
        pos_labels = [0, 2, 4, 6, 8]
        priors = [0.5, 0.3, 0.7]
        train_size = (2500, 50000)
        validation_size = (500, 10000)
        max_epochs = 150
        batch_size = 500
        stepsize = 1e-5
    elif dataset_name == "fmnist":
        pos_labels = [0, 1, 7, 8]
        priors = [0.4, 0.2, 0.6]
        train_size = (2500, 50000)
        validation_size = (500, 10000)
        max_epochs = 150
        batch_size = 500
        stepsize = 1e-5
    
    run(device_num=device_num,
        trial=trial,
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
        path=path
    )


if __name__ == '__main__':
    main(sys.argv[1:])

