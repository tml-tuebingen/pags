# pags
Which Models have Perceptually-Aligned Gradients? An Explanation via Off-Manifold Robustness (Neurips 2023)


train_models.py : contains code to train deep learning classifiers with various kinds of regularization, with some minor logging / checkpointing functionality

utils/regularized_loss.py : defines the different regularized losses we are interested in for training (smooth) classifiers. Currently we only have gradient norm regularization and Gaussian noise smoothness. Can add adversarial training later on to this.

experiments/* : folder describing various experiments (TIP: run with python -m experiments.estimate_smoothness)

run_exp_harvard.py : Harvard-specific code to run jobs on the grid using SLURM

models/* : contains definitions for different models we are interested in, accessed via models/model_selector.py

utils/*: random housekeeping things like logging, paths, datasets
