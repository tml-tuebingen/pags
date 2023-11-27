"""
Entry point to run the different experiments on the internal cluster
"""

import sys
import os
from itertools import product
import numpy as np


def dict2options(settings):
    keys = []
    opts = []
    for k in settings:
        keys.append(k)
        opts.append(settings[k])

    exp_list = list(product(*opts))

    options_str = []
    name_str = []
    for option in exp_list:
        temp_str = ""
        temp_name_str = ""
        for k, i in zip(keys, option):
            temp_str += f"--{k}={i} "
            if ("path" not in k) and ("filename" not in k):
                temp_name_str += f"{i}_"
        options_str.append(temp_str)
        name_str.append(temp_name_str[0:-1])

    return options_str, name_str


def run(filename: str, options: str, path: str = None):
    # we are already in the slurm job, simply run the python file
    if path is None:
        full_cmd = f"python {filename} {options}"
    else:
        full_cmd = f"cd {path} && python {filename} {options}"
    os.system(full_cmd)


def train_standard_models(slurm_task_id):
    # draw a random seed for different replications
    seed = np.random.randint(100000)
    settings = {
        "model-name": [f"'resnet18_seed={seed}_cifar10.pt'"],
        "model-arch": ["resnet18"],
        "regularizer": ["none"],
        "dataset": ["cifar10"],
        "num-epochs": ["200"],
        "save-path": ["saved_models/cifar10_standard"],
        "lr": ["0.025"],
        "prng_seed": [seed],
    }
    options_str, name_str = dict2options(settings)
    all_tasks = list(zip(options_str, name_str))
    print("Total number of tasks:", len(all_tasks))
    print(all_tasks[slurm_task_id])
    run("train_models.py", all_tasks[slurm_task_id][0])


def train_gnorm_regularized(slurm_task_id):
    # draw a random seed for different replications
    seed = np.random.randint(100000)
    # scale the learning rate with the regularization constant
    reg_constant = np.logspace(-5, 2, 30)[slurm_task_id]
    lr = 0.025
    if reg_constant > 0.01:
        lr = 0.025 / (reg_constant / 0.01)
    # create the settings and run job
    settings = {
        "model-name": [
            f"'resnet18_reg=gnorm_const={reg_constant}_seed={seed}_cifar10.pt'"
        ],
        "model-arch": ["resnet18"],
        "regularizer": ["gnorm"],
        "dataset": ["cifar10"],
        "num-epochs": ["200"],
        "save-path": ["saved_models/cifar10_gnorm_replications/"],
        "prng_seed": [seed],
        "reg-constant": [reg_constant],
        "lr": [lr],
    }
    options_str, name_str = dict2options(settings)
    all_tasks = list(zip(options_str, name_str))
    task = all_tasks[0][0]
    print(task)

    run("train_models.py", task)


def train_smooth_regularized(slurm_task_id):
    # draw a random seed for different replications
    seed = np.random.randint(100000)
    # scale the learning rate with the regularization constant
    reg_constant = np.logspace(-4, 4, 30)[slurm_task_id]
    lr = 0.025
    if reg_constant > 10:
        lr = 0.025 / (reg_constant / 10)
    # create the settings and run job
    settings = {
        "model-name": [
            f"'resnet18_reg=gnorm_const={reg_constant}_seed={seed}_cifar10.pt'"
        ],
        "model-arch": ["resnet18"],
        "regularizer": ["smooth"],
        "dataset": ["cifar10"],
        "num-epochs": ["200"],
        "noise-std": [0.1],
        "save-path": ["saved_models/cifar10_smooth_replications/"],
        "prng_seed": [seed],
        "reg-constant": [reg_constant],
        "lr": [lr],
    }
    options_str, name_str = dict2options(settings)
    all_tasks = list(zip(options_str, name_str))
    task = all_tasks[0][0]
    print(task)

    run("train_models.py", task)


def train_randomized_smoothing(slurm_task_id):
    # draw a random seed for different replications
    seed = np.random.randint(100000)
    # choose the noise level
    noise_level = np.logspace(-3, 2, 30)[slurm_task_id]
    # create the settings and run job
    settings = {
        "model-name": [
            f"'resnet18_reg=rand_smooth_noise_level={noise_level}_seed={seed}_cifar10.pt'"
        ],
        "model-arch": ["resnet18"],
        "regularizer": ["rand_smooth"],
        "dataset": ["cifar10"],
        "num-epochs": ["200"],
        "noise-std": [noise_level],
        "save-path": ["saved_models/cifar10_randomized_smoothing_replications/"],
        "prng_seed": [seed],
        "lr": [0.025],
    }
    options_str, name_str = dict2options(settings)
    all_tasks = list(zip(options_str, name_str))
    task = all_tasks[0][0]
    print(task)

    run("train_models.py", task)


def train_pgd(slurm_task_id):
    # draw a random seed for different replications
    seed = np.random.randint(100000)
    # choose the noise level
    # epsilon = np.array([0.0, 0.01, 0.5, 1.0, 5., 10., 100., 1000., 10000, 100000])[slurm_task_id]
    epsilon = np.logspace(-5, 4, 30)[slurm_task_id]
    # scale the learning rate for very large perturbations
    lr = 0.025
    if epsilon > 100:
        lr = 0.025 / (epsilon / 100)
    # create the settings and run job
    settings = {
        "model-name": [f"'resnet18_reg=pgd_epsilon={epsilon}_seed={seed}_cifar10.pt'"],
        "model-arch": ["resnet18"],
        "regularizer": ["pgd"],
        "dataset": ["cifar10"],
        "num-epochs": ["200"],
        "epsilon": [epsilon],
        "num_iter": [10],
        "save-path": ["saved_models/cifar10_pgd_replications/"],
        "prng_seed": [seed],
        "lr": [lr],
    }
    options_str, name_str = dict2options(settings)
    all_tasks = list(zip(options_str, name_str))
    task = all_tasks[0][0]
    print(task)

    run("train_models.py", task)


def measure_score_alignment(slurm_task_id):
    # create the settings and run job
    settings = {
        "model-list-filename": [
            "../saved_models/cifar10_randomized_smoothing_replications.txt"
        ],  # cifar10_gnorm_replications
        "out-folder": ["results/cifar10_randomized_smoothing_score_alignment_lpips"],
        "dataset": ["cifar10"],
    }
    options_str, name_str = dict2options(settings)
    all_tasks = list(zip(options_str, name_str))
    task = all_tasks[0][0]
    print(task)

    run("measure_score_alignment.py", task, "experiments")


def measure_manifold_robustness(slurm_task_id):
    # create the settings and run job
    settings = {
        "model-list-filename": [
            "../saved_models/cifar10_randomized_smoothing_replications.txt"
        ],
        "out-folder": ["results/cifar10_randomized_smoothing_manifold_robustness"],
    }
    options_str, name_str = dict2options(settings)
    all_tasks = list(zip(options_str, name_str))
    task = all_tasks[0][0]
    print(task)

    run("measure_manifold_robustness.py", task, "experiments")


def create_imagenet64x64():
    import torch, torchvision, tqdm
    from torchvision import transforms

    imagenet_train_64x64 = torchvision.datasets.ImageNet(
        "/scratch_local/datasets/ImageNet2012",
        split="train",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(64),
                transforms.ToTensor(),
            ]
        ),
    )

    imagenet_val_64x64 = torchvision.datasets.ImageNet(
        "/scratch_local/datasets/ImageNet2012",
        split="val",
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(64),
                transforms.ToTensor(),
            ]
        ),
    )

    trainloader_64x64 = torch.utils.data.DataLoader(
        imagenet_train_64x64, batch_size=512, shuffle=False, num_workers=30
    )
    valloader_64x64 = torch.utils.data.DataLoader(
        imagenet_val_64x64, batch_size=512, shuffle=False, num_workers=30
    )

    train_images = []
    train_labels = []
    for img, label in tqdm.tqdm(trainloader_64x64):
        train_images.append((img * 255).to(torch.uint8))
        train_labels.append(label)
    train_images = torch.vstack(train_images)
    train_labels = torch.hstack(train_labels)
    torch.save((train_images, train_labels), "datasets/imagenet-64x64-train.pth")

    val_images = []
    val_labels = []
    for img, label in tqdm.tqdm(valloader_64x64):
        val_images.append((img * 255).to(torch.uint8))
        val_labels.append(label)
    val_images = torch.vstack(val_images)
    val_labels = torch.hstack(val_labels)
    torch.save((val_images, val_labels), "datasets/imagenet-64x64-val.pth")


def train_imagenet64x64(slurm_task_id):
    epsilons = [0.0, 0.01, 0.1, 5, 10, 20, 50, 100, 200, 500, 2500, 5000]

    run("train_robust_imagenet.py", f"--epsilon {epsilons[slurm_task_id]} --resume")


if __name__ == "__main__":
    slurm_task_id = int(sys.argv[1])
    print(f"SLURM task id: {slurm_task_id}")

    create_imagenet64x64(slurm_task_id)
    train_imagenet64x64(slurm_task_id)

    train_standard_models(slurm_task_id)
    train_gnorm_regularized(slurm_task_id)
    train_smooth_regularized(slurm_task_id)
    train_randomized_smoothing(slurm_task_id)
    train_pgd(slurm_task_id)

    measure_score_alignment(slurm_task_id)
    measure_manifold_robustness(slurm_task_id)
