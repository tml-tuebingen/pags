import utils.dnnlib
import torch
import pickle

from captum.attr import Saliency

###############################################################################################################################
#        Wrapping the code from "Elucidating the Design Space of Diffusion-Based Generative Models (EDM)", NeurIPS 2022
###############################################################################################################################
model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
cifar10_network_pkl = f"{model_root}/edm-cifar10-32x32-uncond-vp.pkl"
CIFAR10_MODEL = None


def cifar10_score(images: torch.Tensor, sigma: float, device=torch.device("cuda")):
    """Estimate the score of the cifar10 data distribution convoluted with normal noise of standard deviation sigma, using the diffusion model from the 2022 neurips paper."""
    global CIFAR10_MODEL
    # load model
    if CIFAR10_MODEL is None:
        print(f'Loading network from "{cifar10_network_pkl}"...')
        with utils.dnnlib.open_url(cifar10_network_pkl) as f:
            CIFAR10_MODEL = pickle.load(f)["ema"].to(device)
    # pass images through the model
    return edm_score(images, CIFAR10_MODEL, sigma, device=device)


def edm_score(
    images: torch.Tensor,
    diffusion_model: torch.nn.Module,
    sigma: float,
    class_labels=None,
    device=torch.device("cuda"),
    scale_images=True,
):
    # check parameters
    if sigma <= 0:
        raise ValueError("Noise level sigma has to be in (0,+INF)")
    if len(images.shape) != 4 or images.shape[2] != images.shape[3]:
        raise ValueError(
            f"Expected images to be a tensor of shape (batch_size, 3, n,n), but has shape {images.shape} instead."
        )
    # scale images from [0,1] to [-1,1]
    if scale_images:
        if images.min() < 0 or images.max() > 1:
            print(
                "WARNING: edm_score expected images in [0,1]. Estimated score is probably inaccurate."
            )
        images = 2 * images - 1
    else:
        if images.min() >= 0 or images.max() > 1:
            print(
                "WARNING: edm_score expected images in [-1,1]. Estimated score is probably inaccurate."
            )
        print("Done")
    # proper device
    diffusion_model.to(device)
    images = images.to(device)
    sigma = diffusion_model.round_sigma(torch.Tensor([sigma]).to(device))
    # format class_labels
    if diffusion_model.label_dim == 0:
        if class_labels is not None:
            print("WARNING: edm_score: class_scores with unconditional model")
        class_labels = None
    if class_labels is not None:
        batch_size = class_labels.shape[0]
        class_labels_matrix = torch.zeros(
            (batch_size, diffusion_model.label_dim), dtype=torch.int64, device=device
        )
        for idx in range(batch_size):
            class_labels_matrix[idx, class_labels[idx]] = 1
        class_labels = class_labels_matrix
    # pass through the neural network
    D = diffusion_model(images.to(torch.float32), sigma, class_labels).to(images.dtype)
    # compute the score
    score = (D - images) / sigma.item() ** 2
    # return result
    score = score.detach().cpu()
    return score


###############################################################################################################################
#                                                Compute Inpute Gradients
###############################################################################################################################


def input_gradient(model, img, target=None, device="cuda"):
    """Returns the gradient with respect to the input."""
    img = img.to(device)
    model.to(device)
    if target is None:
        target = model(img).argmax(dim=1)
    saliency = Saliency(model)
    img.requires_grad = True
    return saliency.attribute(img, target=target, abs=False).detach().cpu()


def input_gradient_sum(model, img, device="cuda"):
    """The sum of the input gradients for all different classes"""
    img = img.to(device)  # prepare
    img.requires_grad = True
    model.to(device)
    model.zero_grad()
    # compute
    # model(img).sum(dim=1).backward()
    model(img).sum().backward()
    result = img.grad.detach().cpu()
    # clean-up
    model.zero_grad()
    img.grad = None
    return result
