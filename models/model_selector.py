from functools import partial

import torch.nn.utils.parametrizations
import torch
import models.resnet
import models.vgg
import models.diffusion_model
import models.lenet

# Which base model to load?
cifar_models = {
    "resnet18": partial(models.resnet.ResNet18, activation=torch.nn.ReLU),
    "resnet34": partial(models.resnet.ResNet34, activation=torch.nn.ReLU),
    "vgg11": partial(models.vgg.VGG, vgg_name="VGG11", activation=torch.nn.ReLU),
    "resnet18softplus": partial(
        models.resnet.ResNet18, activation=partial(torch.nn.Softplus, beta=20)
    ),
    "resnet34softplus": partial(
        models.resnet.ResNet34, activation=partial(torch.nn.Softplus, beta=20)
    ),
    "vgg11softplus": partial(
        models.vgg.VGG, vgg_name="VGG11", activation=partial(torch.nn.Softplus, beta=20)
    ),
}

mnist_models = {"lenet": partial(models.lenet.LeNet)}

imagenet_models = {"resnet18": None}

base_models = {
    "cifar10": cifar_models,
    "cifar100": cifar_models,
    "mnist": mnist_models,
    "word_distractor_mnist": {
        "resnet18": partial(
            models.resnet.ResNet18, activation=torch.nn.ReLU, in_channel=1
        ),
    },
    "simple_word_distractor_mnist": {
        "resnet18": partial(
            models.resnet.ResNet18, activation=torch.nn.ReLU, in_channel=1
        ),
    },
    "imagenet64": imagenet_models,
}


def model_architecture(
    model_name: str, num_classes: int = 10, dataset: str = "cifar10"
):
    return base_models[dataset][model_name](num_classes)


if __name__ == "__main__":
    batch_size = 5
    classes = 10

    # Testing MNIST
    model_name = "lenet"
    model = model_architecture(model_name, num_classes=classes, dataset="mnist")
    random_input = torch.rand((batch_size, 1, 28, 28))
    torch.testing.assert_close(
        model(random_input).size(), torch.Size([batch_size, classes])
    )
