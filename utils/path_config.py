import os
from typing import Dict


def get_paths() -> Dict[str, str]:
    pwd = os.getcwd()

    # Parent directory
    data_home = os.path.dirname(pwd) + "/datasets"
    cifar10 = data_home + "/cifar10"
    svhn = data_home + "/svhn"
    mnist = data_home + "/mnist"
    word_distractor_mnist = data_home + "/word_distractor_mnist/"
    saved_models = os.path.join(pwd, "saved_models")
    logs = os.path.join(pwd, "logs")
    paths = {
        "cifar10": cifar10,
        "pwd": pwd,
        "logs": logs,
        "saved_models": saved_models,
        "mnist": mnist,
        "word_distractor_mnist": word_distractor_mnist,
        "simple_word_distractor_mnist": word_distractor_mnist,
    }
    return paths


if __name__ == "__main__":
    paths = get_paths()
