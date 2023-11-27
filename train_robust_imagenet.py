"""
Train robust models with projected gradient descent on ImageNet64x64.
"""

import torch
from torch import nn

import numpy as np
import tqdm
import os
import sys

from torchvision import transforms
from torchvision.models import resnet18
from collections import OrderedDict

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def pgd_l2(model, X, y, epsilon=[5.0], num_iter=3):
    """Construct FGSM adversarial examples on the examples X"""
    # randomly choose epsilon
    epsilon = np.random.choice(epsilon)

    # choose a random starting point with length epsilon / 2
    delta = torch.rand_like(X, requires_grad=True)
    norm = torch.linalg.norm(delta.flatten())
    delta.data = epsilon * delta.data / norm / 2

    alpha = 2 * epsilon / num_iter  # fixed step size of 2 * epsilon/ num_iter
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()

        # take a step
        step = delta.grad.detach()
        step = alpha * step / torch.linalg.norm(step.flatten())
        delta.data = delta.data + step

        # project on the epsilon ball around X if necessary
        norm = torch.linalg.norm(delta.flatten())
        if norm > epsilon:
            delta.data = epsilon * delta.data / norm

        # next iteration
        delta.grad.zero_()
    return delta.detach()


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment arguments")

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Adversarial Perturbation Budget",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint (if it exists)",
    )

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    args = parse_args()

    # data
    train_images, train_labels = torch.load("datasets/imagenet-64x64-train.pth")
    val_images, val_labels = torch.load("datasets/imagenet-64x64-val.pth")

    trainset = torch.utils.data.TensorDataset(train_images, train_labels)
    valset = torch.utils.data.TensorDataset(val_images, val_labels)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip()]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4096, shuffle=True, num_workers=8
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=4096, shuffle=False, num_workers=8
    )

    # model
    model = resnet18()

    # find latest checkpoint and resume training from there
    i_checkpoint_epoch = 0
    if args.resume:
        final_file = f"saved_models/imagenet_robust/imagenet64x64/resnet18_l2_eps{args.epsilon}.pth"
        if os.path.isfile(final_file):
            print("Training is already complete")
            sys.exit("Training is already complete")
        for i_checkpoint_epoch in reversed(range(90)):
            checkpoint = f"saved_models/imagenet_robust/imagenet64x64/resnet18_l2_eps{args.epsilon}_epoch{i_checkpoint_epoch}.pth"
            if os.path.isfile(checkpoint):
                print(f"Resuming from checkpoint {checkpoint}")
                state_dict = torch.load(checkpoint)
                # remove `module.` from distributed training
                if "module.conv1.weight" in state_dict.keys():
                    cleaned_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]
                        cleaned_state_dict[name] = v
                    state_dict = cleaned_state_dict
                model.load_state_dict(state_dict)
                break
    if i_checkpoint_epoch == 89:  # no checkpoint found
        i_checkpoint_epoch = 0

    # use data parallel for trainin on multiple gpus
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), " GPUs")
        model = torch.nn.DataParallel(model)

    # training setup
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    for epoch in range(i_checkpoint_epoch):  # a bit hacky, but it works
        scheduler.step()
    ce_loss = torch.nn.CrossEntropyLoss()

    for i_epoch in range(i_checkpoint_epoch + 1, 90):
        # training
        model.train()
        train_loss = 0
        train_zero_one_loss = 0
        for img, label in tqdm.tqdm(trainloader):
            img = img / 255
            img = train_transform(img)
            img, label = img.to(device), label.to(device)
            if args.epsilon > 0:  # adversarial training
                delta = pgd_l2(model, img, label, epsilon=[args.epsilon])
                pred = model(img + delta)
            else:  # standard training
                pred = model(img)
            optimizer.zero_grad()
            loss = ce_loss(pred, label)
            loss.backward()
            train_loss += loss.item()
            train_zero_one_loss += (
                (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()
            )
            optimizer.step()

        # validation
        model.eval()
        val_loss = 0
        val_zero_one_loss = 0
        for img, label in tqdm.tqdm(valloader):
            img = img / 255
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss = ce_loss(pred, label)
            val_loss += loss.item()
            val_zero_one_loss += (
                (pred.softmax(dim=1).argmax(dim=1) != label).sum().item()
            )

        print(
            f"Epoch {i_epoch}. Loss: {train_loss / len(trainloader.dataset)}. Val Loss: {val_loss / len(valloader.dataset)}. \
                Acc.: {1-train_zero_one_loss / len(trainloader.dataset)}.  Val Acc. {1-val_zero_one_loss / len(valloader.dataset)}"
        )
        scheduler.step()

        # checkpoint the model after each epoch
        torch.save(
            model.state_dict(),
            f"saved_models/imagenet_robust/imagenet64x64/resnet18_l2_eps{args.epsilon}_epoch{i_epoch}.pth",
        )
        prev_file = f"saved_models/imagenet_robust/imagenet64x64/resnet18_l2_eps{args.epsilon}_epoch{i_epoch-1}.pth"
        if os.path.isfile(prev_file):
            os.remove(prev_file)

    torch.save(
        model.state_dict(),
        f"saved_models/imagenet_robust/imagenet64x64/resnet18_l2_eps{args.epsilon}.pth",
    )
