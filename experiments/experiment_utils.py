""" 
Some utility functions that are used by all experiments 
"""

import random
import os
import logging

import torch

import utils
import utils.logging
import utils.path_config as path_config

import models.model_selector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(utils.logging.get_standard_streamhandler())


def load_model(args, modelname, num_classes, device):
    """Load a model from file"""
    model = models.model_selector.model_architecture(
        args.model_arch, num_classes, args.dataset
    )
    model.load_state_dict(torch.load(modelname), strict=True)
    model.to(device)
    model.eval()
    return model


def seed_data_device(args):
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

    paths = path_config.get_paths()

    # Load Datasets
    logger.info(f"Loading datasets")
    os.makedirs(paths[args.dataset], exist_ok=True)
    train_loader, test_loader = utils.datasets.get_dataloaders(
        args.dataset, args.batch_size
    )

    return train_loader, test_loader, device
