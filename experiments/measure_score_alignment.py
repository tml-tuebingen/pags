"""
Measure the alignment of a model with the score of the probability distribution.
"""
import os

import logging

import argparse
import socket
import datetime as dt
from pathlib import Path
import pickle

import torch

import lpips

import utils.logging
import utils.datasets

from utils.edm_score import input_gradient_sum, cifar10_score

from experiment_utils import seed_data_device, load_model

# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(utils.logging.get_standard_streamhandler())


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
        default="results/score_alignment_lpips",
        help="Output foldername to store results",
    )

    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "svhn", "mnist", "imagenet"],
        default="cifar10",
        help="Which dataset to use?",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Sigma used to estimate the score",
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for estimation",
    )

    parser.add_argument("--batch-size", type=int, default=64)

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

    # compute the score for all images from the test set
    sigma = args.sigma

    images = []
    scores = []
    for img, _ in test_loader:
        images.append(img.detach().cpu())
        scores.append(
            cifar10_score(img.to(device), sigma, device=device).detach().cpu()
        )
    images = torch.vstack(images)
    scores = torch.vstack(scores)

    # scale the score so that it lies in [-1,1]
    for idx in range(scores.shape[0]):
        scores[idx] = scores[idx] / scores[idx].abs().max()

    # number of images to use (data-fraction parameter)
    N_images = int(args.data_fraction * images.shape[0])
    print(f"Estimating with a total of {N_images} images.")

    # init lpips
    torch.hub.set_dir("../tmp/.cache/torchhub")  # set hub to writeable directory
    loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores

    for modelname in model_list:  # for all models
        # file to store results
        fout = f"{args.out_folder}/score_alignment_{os.path.basename(modelname)}.pkl"
        print(f"Storing results in {fout}")

        # Load Model
        model = load_model(args, modelname, num_classes, device)
        logger.info(f"Model:{modelname}")

        # compute input gradients
        input_gradients = []
        for idx, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)
            # gradient = input_gradient(model, img, label).detach().cpu() # with conditional diffusion model
            gradient = (
                input_gradient_sum(model, img).detach().cpu()
            )  # with unconditional diffusion model
            input_gradients.append(gradient)
            if args.batch_size * idx > N_images:
                break
        input_gradients = torch.cat(input_gradients)

        # scale input gradients so that they lie in [-1,1]
        for idx in range(input_gradients.shape[0]):
            input_gradients[idx] = (
                input_gradients[idx] / input_gradients[idx].abs().max()
            )

        # compute the LPIPS distance between the input gradients and the scores
        distances = []
        for i_img in range(N_images):  # for all images
            distance = loss_fn_alex(
                input_gradients[i_img : i_img + 1], scores[i_img : i_img + 1]
            )
            distances.append(distance.item())

        # store results
        with open(fout, "wb") as pickle_file:
            pickle.dump(distances, pickle_file)
