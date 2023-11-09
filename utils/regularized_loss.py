import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

import numpy as np


class CELoss(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_criterion: Callable = torch.nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.metadata = {}
        self.model = model
        self.loss_criterion = loss_criterion

    def compute_loss(self, input, target):
        out = self.model(input)
        raw_loss = self.loss_criterion(out, target)
        self.metadata = {"raw_loss": raw_loss.item()}
        return raw_loss


class GradNormRegularizedLoss(CELoss):
    # norm of gradient of out w.r.t. input
    # usually out are logits or log-probabilites
    def __init__(self, model: nn.Module, reg_constant: float = 1e-3):
        super().__init__(model)
        self.reg_constant = reg_constant

    def compute_loss(self, input, target):
        out = self.model(input)
        raw_loss = self.loss_criterion(out, target)

        grad_x = torch.autograd.grad(
            raw_loss, input, only_inputs=True, create_graph=True
        )[0]
        gradnorm = 1e5 * grad_x.pow(2).sum() / input.size(0)

        self.metadata = {"raw_loss": raw_loss.item(), "gradnorm": gradnorm.item()}
        return raw_loss + self.reg_constant * gradnorm


class NoiseRegularizedLoss(CELoss):
    # add noise to inputs, and
    # ensure f(x+noise) ~ f(x)
    def __init__(
        self, model: nn.Module, reg_constant: float = 1e-3, noise_std: float = 1e-2
    ):
        super().__init__(model)
        self.reg_constant = reg_constant
        self.noise_std = noise_std

    def compute_loss(self, input, target):
        out = self.model(input)
        raw_loss = self.loss_criterion(out, target)

        noise = torch.normal(mean=torch.zeros_like(input), std=self.noise_std)
        out_noisy = self.model(input + noise)

        reg_loss = torch.pow(out - out_noisy, 2).sum() / input.size(0)

        self.metadata = {
            "raw_loss": raw_loss.item(),
            "smoothness_loss": reg_loss.item(),
        }
        return raw_loss + self.reg_constant * reg_loss


class RandomizedSmoothingLoss(CELoss):
    # add noise to inputs, i.e. classify f(x+noise)
    def __init__(self, model: nn.Module, noise_std: float):
        super().__init__(model)
        self.noise_std = noise_std

    def compute_loss(self, input, target):
        noise = torch.normal(mean=torch.zeros_like(input), std=self.noise_std)
        out = self.model(input + noise)

        loss = self.loss_criterion(out, target)

        self.metadata = {
            "loss": loss.item(),
        }
        return loss


class PGDAdversarialLoss(CELoss):
    # projected gradient descent (PGD) with l2 metric
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.5,
        num_iter: int = 15,
        loss_criterion: Callable = torch.nn.CrossEntropyLoss(),
    ):
        super().__init__(model)
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.loss_criterion = loss_criterion

    def compute_loss(self, input, target):
        delta = self.pgd_l2_(self.model, input, target, self.epsilon, self.num_iter)
        out = self.model(input + delta)

        loss = self.loss_criterion(out, target)

        self.metadata = {
            "adv_loss": loss.item(),
        }
        return loss

    def pgd_l2_(self, model, X, y, epsilon, num_iter):
        """Construct PGD adversarial examples on the examples X"""

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
