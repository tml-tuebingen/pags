import os
import argparse
import contextlib

import torch
import utils.misc as misc
import utils.path_config as path_config


def args_to_modelname(args: argparse.Namespace) -> str:
    if args.model_name is None:
        model_filename = args.model_arch
        if args.regularizer is not None:
            model_filename += f"_reg={args.regularizer}"
            if args.regularizer == "smooth":
                model_filename += f"_const={args.reg_constant}_std={args.noise_std}"
            if args.regularizer == "gnorm":
                model_filename += f"_const={args.reg_constant}"
            if args.model_arch[:5] == "diff_":
                model_filename += f"_diff_sigma={args.diffusion_sigma}"
        if args.dataset is not None:
            model_filename += f"_{args.dataset}"
    else:
        model_filename = args.model_name
    return model_filename


class ModelLogging:
    # Save trained and partially trained models in a directory
    def __init__(self, model_filename: str, save_path: str, logger, device) -> None:
        self.device = device
        self.logger = logger

        model_fullfilename = os.path.join(save_path, model_filename)
        self.model_fullfilename = model_fullfilename
        os.makedirs(save_path, exist_ok=True)

        logger.info(f"Starting run -- checkpoints will be at {model_fullfilename}")

    def save_model_and_checkpoint(self, epoch, num_epochs, model):
        # Save the current model
        if epoch < num_epochs:
            epoch_fullfilename = f"{self.model_fullfilename}_{epoch}.pt"
        else:
            epoch_fullfilename = f"{self.model_fullfilename}.pt"

        self.logger.debug(f"Saving to {self.model_fullfilename}")
        torch.save(model.cpu().state_dict(), epoch_fullfilename)

        # delete previously stored model
        if epoch >= 1:
            with contextlib.suppress(FileNotFoundError):
                os.remove(f"{self.model_fullfilename}_{epoch - 1}.pt")
        model.to(self.device)
