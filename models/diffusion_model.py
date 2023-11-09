# Wrapping the code from "Elucidating the Design Space of
# Diffusion-Based Generative Models (EDM)", NeurIPS 2022
# to look like a pytorch module

import torch
import pickle

from utils.dnnlib import open_url
from utils.edm_score import cifar10_score

nvidia_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"

# Which diffusion model to load?
network_pickle = {
    "cifar10": f"{nvidia_root}/edm-cifar10-32x32-uncond-vp.pkl",
    "imagenet64": f"{nvidia_root}/edm-imagenet-64x64-cond-adm.pkl",
    "svhn": None,
}


class Diffusion(torch.nn.Module):
    def __init__(self, network_pickle, scale_images=True, verbose=False) -> None:
        super().__init__()
        self.scale_images = scale_images
        self.verbose = verbose
        # load model
        print(f'Loading network from "{network_pickle}"...')
        with open_url(network_pickle) as f:
            self.model = pickle.load(f)["ema"]
        print("Done")

    def _scale(self, image: torch.Tensor):
        if self.scale_images:
            if self.verbose and (image.min() < 0 or image.max() > 1):
                print(
                    "WARNING: expected images in [0,1]. \
                    Estimated score might be inaccurate."
                )
            image = 2 * image - 1
        else:
            if self.verbose and (image.min() >= 0 or image.max() > 1):
                print(
                    "WARNING: expected images in [-1,1]. \
                    Estimated score is probably inaccurate."
                )
        return image

    def _format_class_labels(self, class_labels):
        """With conditional diffusion models, we must pass the class labels in order to estimate the score.
        The edm diffusion models expects class labels in a matrix format. This function brings class labels into that format.
        """
        if self.model.label_dim == 0:
            if class_labels is not None:
                print("WARNING: edm_score: class_scores with unconditional model")
            class_labels = None
        if class_labels is not None:
            batch_size = class_labels.shape[0]
            class_labels_matrix = torch.zeros(
                (batch_size, self.model.label_dim),
                dtype=torch.int64,
                device=class_labels.device,
            )
            for idx in range(batch_size):
                class_labels_matrix[idx, class_labels[idx]] = 1
            class_labels = class_labels_matrix
        return class_labels

    def get_score(self, image: torch.Tensor, sigma: float, class_labels=None):
        image = self._scale(image)
        class_labels = self._format_class_labels(class_labels)
        if class_labels is not None:
            class_labels = class_labels.to(image.device)

        sigma = self.model.round_sigma(torch.Tensor([sigma]))
        D = self.model(
            image.to(torch.float32),
            sigma.to(image.device),
            class_labels,
        ).to(image.dtype)
        # compute the score
        score = (D - image) / sigma.item() ** 2
        return score

    def forward(self, image: torch.Tensor, sigma: float):
        image = self._scale(image)  # scale the images from [0,1] to [-1,1]

        sigma = self.model.round_sigma(torch.Tensor([sigma]))
        diffusion_out = self.model(image.to(torch.float32), sigma.to(image.device)).to(
            image.dtype
        )
        if self.scale_images:
            return (diffusion_out + 1) / 2  # scale the images from [-1,1] to [0,1]
        return diffusion_out


def select_diffusion_model(dataset_name: str):
    """obtain the diffusion model for some dataset"""
    if not dataset_name in network_pickle:
        raise ValueError(f"Do not know about dataset = {dataset_name}")
    if network_pickle[dataset_name] is None:
        raise ValueError(f"No diffusion model for dataset = {dataset_name}")
    return Diffusion(network_pickle[dataset_name])


def test_equivalence_with_cifar10_score():
    # testing whether it gives same results as the code in edm_score.cifar10_score
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    random_input = torch.rand((5, 3, 32, 32))
    diff = Diffusion(network_pickle=network_pickle["cifar"]).to(device)
    module_score = diff.get_score(random_input, sigma=0.1)
    edm_score = cifar10_score(random_input, sigma=0.1, device=device)

    torch.testing.assert_close(module_score, edm_score)


if __name__ == "__main__":
    test_equivalence_with_cifar10_score()
