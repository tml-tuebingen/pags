"""
Measure the on- and off-manifold robustness of a model.
"""
import os

import numpy as np
import logging
import argparse
import socket
import datetime as dt
from pathlib import Path
import pickle

from scipy.linalg import orth

import torch
import torch.nn.functional as F

import utils.logging
import utils.datasets

from experiment_utils import seed_data_device, load_model

# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(utils.logging.get_standard_streamhandler())


def softmax_l1(x, y):
    return (F.softmax(x, dim=1) - F.softmax(y, dim=1)).abs().sum(axis=1)


def project_into_tangent_space(tangent_space, vector):
    dim = tangent_space.shape[0]
    batch_dim = tangent_space.shape[1]
    img_dim = tangent_space.shape[2]
    coeff = np.zeros(dim)
    tangent_space_orth = orth(
        tangent_space.reshape((-1, batch_dim * img_dim * img_dim)).T
    ).T.reshape((-1, batch_dim, img_dim, img_dim))
    for i in range(dim):
        coeff[i] = tangent_space_orth[i, :, :].flatten() @ vector.flatten()
    vector_in_tangent_space = (coeff @ tangent_space_orth.reshape((dim, -1))).reshape(
        (batch_dim, img_dim, img_dim)
    )
    return vector_in_tangent_space


def measure_on_off_manifold_robustness(
    model, test_loader, test_tangent_spaces, device, N_images
):
    step_sizes = np.logspace(-3, 2, num=20)[12:16]

    on_manifold_robustness = {epsilon: [] for epsilon in step_sizes}
    off_manifold_robustness = {epsilon: [] for epsilon in step_sizes}

    for idx, (img, _) in enumerate(test_loader):
        logits = model(img.to(device)).detach()
        # draw random on- and off- manifold directions
        noise = torch.randn_like(img.squeeze())
        tangent_noise = project_into_tangent_space(
            test_tangent_spaces[idx].numpy(), noise.numpy()
        )
        ortho_noise = noise - tangent_noise
        # normalize, so we can make epsilon steps
        tangent_noise = tangent_noise / np.linalg.norm(tangent_noise)
        ortho_noise = ortho_noise / np.linalg.norm(ortho_noise)
        # measure change in model output
        for epsilon in step_sizes:
            # on-manifold
            epsilon_logits = model(
                (img + epsilon * tangent_noise).to(torch.float32).to(device)
            ).detach()
            on_manifold_robustness[epsilon].append(
                (logits.detach().cpu(), epsilon_logits.detach().cpu())
            )
            # off-manifold
            epsilon_logits = model(
                (img + epsilon * ortho_noise).to(torch.float32).to(device)
            ).detach()
            off_manifold_robustness[epsilon].append(
                (logits.detach().cpu(), epsilon_logits.detach().cpu())
            )
        if idx >= N_images:
            break

    return on_manifold_robustness, off_manifold_robustness


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment arguments")

    parser.add_argument(
        "--model-arch", default="resnet18", help="What architecture to use?"
    )

    parser.add_argument(
        "--model-list-filename",
        type=str,
        help="Path of textfile which has a list of paths of models to evaluate",
    )

    parser.add_argument(
        "--out-folder",
        type=str,
        default="results/manifold_robustness",
        help="Output foldername to store results",
    )

    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "svhn", "mnist", "imagenet"],
        default="cifar10",
        help="Which dataset to use?",
    )

    parser.add_argument(
        "--tangent-spaces",
        default="results/test_tangent_spaces.pt",
        help="The pre-computed tangent spaces",
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for estimation",
    )

    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--prng_seed", type=int, default=1729)

    args = parser.parse_args()

    # Show user some information about current job
    logger.info(f"UTC time {dt.datetime.utcnow():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Host: {socket.gethostname()}")

    logger.info("\n----------------------------")
    logger.info("    Argparse arguments")
    logger.info("----------------------------")
    # print all argparse'd args
    for arg in vars(args):
        logger.info(f"{arg} \t {getattr(args, arg)}")

    logger.info("----------------------------\n")

    return args


if __name__ == "__main__":
    args = parse_args()
    train_loader, test_loader, device = seed_data_device(args)
    num_classes = utils.datasets.get_num_classes(args.dataset)
    os.makedirs(args.out_folder, exist_ok=True)

    # get all models
    model_list = Path(args.model_list_filename).read_text().splitlines()

    # load the pre-computed tangent spaces
    test_tangent_spaces = torch.load(args.tangent_spaces)  # a list of tensors

    # number of images to use (data-fraction parameter)
    N_images = int(args.data_fraction * len(test_tangent_spaces))
    print(f"Estimating with a total of {N_images} images.")

    for modelname in model_list:  # for all models
        # file to store results
        fout = (
            f"{args.out_folder}/manifold_robustness_{os.path.basename(modelname)}.pkl"
        )
        print(f"Storing results in {fout}")

        # Load Model
        model = load_model(args, modelname, num_classes, device)
        logger.info(f"Model:{modelname}")

        on_robustness, off_robustnes = measure_on_off_manifold_robustness(
            model, test_loader, test_tangent_spaces, device, N_images
        )

        # store results
        with open(fout, "wb") as pickle_file:
            pickle.dump((on_robustness, off_robustnes), pickle_file)
