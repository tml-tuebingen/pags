"""
Train models with various flavours of regularization.
"""

import socket
import argparse
import time
import random
import os
import logging
import datetime as dt

from functools import partial
from typing import Callable

import torch

import utils.path_config as path_config
import utils.datasets
import utils.logging
import utils.misc
import models.model_selector

from utils.regularized_loss import (
    CELoss,
    GradNormRegularizedLoss,
    NoiseRegularizedLoss,
    RandomizedSmoothingLoss,
    PGDAdversarialLoss,
)


logging_level = 15
# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
standard_streamhandler = utils.logging.get_standard_streamhandler()
logger.addHandler(standard_streamhandler)
# End Logging


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    penalty: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device).requires_grad_(), y.to(device)
        optimizer.zero_grad()
        model.zero_grad()

        loss_scalar = penalty.compute_loss(x, y)
        metadata = penalty.metadata

        loss_scalar.backward()
        optimizer.step()

        if 0 == batch_idx % 100:
            log_str = "[{:05d}/{:05d} ({:.0f}%)]".format(
                batch_idx * len(x),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
            )
            for k, v in metadata.items():
                log_str += "\t" + k + ": {:.4f}".format(v)
            logger.info(log_str)


def main():
    parser = argparse.ArgumentParser(description="Experiment arguments")

    parser.add_argument(
        "--model-arch", default="resnet18", help="What architecture to use?"
    )

    parser.add_argument("--model-name", help="Name of trained model to save (optional)")

    parser.add_argument(
        "--diffusion-sigma",
        type=float,
        default=0.1,
        help="Noise level for the diffusion model. Applies only to model class diff_*",
    )

    parser.add_argument(
        "--save-path", help="folder in which to save the model (optional)"
    )

    parser.add_argument(
        "--regularizer",
        choices=["gnorm", "smooth", "rand_smooth", "pgd", "none"],
        help="Which regularizer to use? Default: No regularizer",
    )

    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")

    parser.add_argument(
        "--dataset",
        choices=utils.datasets.get_dataset_names(),
        default="cifar100",
        help="Which dataset to use?",
    )

    parser.add_argument("--num-epochs", type=int, default=200)

    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="initial learning rate (will be modified by scheduler)",
    )

    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum to use w/ sgd (ignored for adam)",
    )

    parser.add_argument(
        "--reg-constant",
        type=float,
        default=1e-3,
        help="smoothness regularization strength",
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=1e-2,
        help="noise level used for smoothness regularization",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="adversarial pertubation budget. for pgd training",
    )

    parser.add_argument(
        "--num_iter",
        type=int,
        default=15,
        help="number of interations for pgd training.",
    )

    parser.add_argument(
        "--prng_seed",
        type=int,
        default=1729,
        help="Use the Ramanujan number as default seed for good luck :P",
    )

    args = parser.parse_args()

    # Show user some information about current job
    logger.info(f"UTC time {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Host: {socket.gethostname()}")

    logger.info("\n------------------------------")
    logger.info("      Argparse arguments")
    logger.info("------------------------------")
    # print all argparse'd args
    for arg in vars(args):
        logger.info(f"{arg} \t {getattr(args, arg)}")

    logger.info("------------------------------\n")
    logger.info(f"torch.__version__ = {torch.__version__}\n")

    return args


def get_model_and_datasets(args):
    # Random seeds
    prng_seed = args.prng_seed
    torch.manual_seed(prng_seed)
    random.seed(prng_seed)

    # Device selection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    if cuda_available:
        torch.cuda.manual_seed(prng_seed)
        torch.backends.cudnn.benchmark = True
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        logger.info(f"Running on {current_device_properties}")
        logger.info(f"torch.version.cuda = {torch.version.cuda}")

    paths = path_config.get_paths()
    os.makedirs(paths["saved_models"], exist_ok=True)

    # Load Datasets
    logger.info(f"Loading datasets")
    os.makedirs(paths[args.dataset], exist_ok=True)
    train_loader, test_loader = utils.datasets.get_dataloaders(
        args.dataset, args.batch_size
    )
    num_classes = utils.datasets.get_num_classes(args.dataset)

    # Load Model
    model = models.model_selector.model_architecture(
        args.model_arch, num_classes, dataset=args.dataset, sigma=args.diffusion_sigma
    ).to(device)

    return model, train_loader, test_loader, device


def get_optimizer_scheduler(args, model):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    # Hard-coded scheduler
    milestone_fracs = [0.750, 0.875]
    gamma = 0.1
    milestones = [int(_ * args.num_epochs) for _ in milestone_fracs]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    return optimizer, scheduler


def get_penalty(args):
    # Various losses as a dictionary
    loss_dict = {
        "gnorm": partial(GradNormRegularizedLoss, reg_constant=args.reg_constant),
        "smooth": partial(
            NoiseRegularizedLoss,
            reg_constant=args.reg_constant,
            noise_std=args.noise_std,
        ),
        "rand_smooth": partial(RandomizedSmoothingLoss, noise_std=args.noise_std),
        "pgd": partial(
            PGDAdversarialLoss, epsilon=args.epsilon, num_iter=args.num_iter
        ),
        "none": partial(CELoss),
    }

    loss_fun = loss_dict.get(args.regularizer, CELoss)
    penalty = loss_fun(model)
    return penalty


if __name__ == "__main__":
    args = main()
    model, train_loader, test_loader, device = get_model_and_datasets(args)
    optimizer, scheduler = get_optimizer_scheduler(args, model)
    penalty = get_penalty(args)

    paths = path_config.get_paths()
    model_filename = utils.misc.args_to_modelname(args)
    save_path = paths["saved_models"] if (args.save_path is None) else args.save_path
    model_logger = utils.misc.ModelLogging(
        model_filename=model_filename, save_path=save_path, logger=logger, device=device
    )

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"Epoch: {epoch:03d} / {args.num_epochs:03d}")
        start = time.time()
        train(model, train_loader, penalty, optimizer, device)
        logger.info("Time: {:.2f} s".format(time.time() - start))

        average_loss, acc = utils.datasets.test(model, test_loader, device)
        logger.info(
            "Test set: Average loss: {:.4f}, Accuracy:({:.0f}%)\n".format(
                average_loss, acc
            )
        )
        scheduler.step()

        model_logger.save_model_and_checkpoint(epoch, args.num_epochs, model)

    logger.info(f"Fitting done. See {model_logger.model_fullfilename}.pt")
