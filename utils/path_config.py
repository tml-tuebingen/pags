import os
from typing import Dict


def get_paths() -> Dict[str, str]:
    pwd = os.getcwd()

    # Parent directory
    data_home = os.path.dirname(pwd) + "/datasets"
    cifar10 = data_home + "/cifar10"
    cifar100 = data_home + "/cifar100/cifar-100-python/"
    svhn = data_home + "/svhn"
    mnist = data_home + "/mnist"
    color_dot_mnist = data_home + "/color_dot_mnist/"
    word_distractor_mnist = data_home + "/word_distractor_mnist/"
    color_dot_cifar = data_home + "/color_dot_cifar/"

    saved_models = os.path.join(pwd, "saved_models")
    logs = os.path.join(pwd, "logs")
    paths = {
        "cifar100": cifar100,
        "cifar10": cifar10,
        "pwd": pwd,
        "svhn": svhn,
        "logs": logs,
        "saved_models": saved_models,
        "mnist": mnist,
        "color_dot_mnist": color_dot_mnist,
        "word_distractor_mnist": word_distractor_mnist,
        "simple_word_distractor_mnist": word_distractor_mnist,
        "color_dot_cifar": color_dot_cifar,
    }
    return paths


if __name__ == "__main__":
    paths = get_paths()
