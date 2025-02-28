# Federated Learning for Pupil Dynamics
### Simon Lübeß (10031581) & Timm Kleipsties (10027833)
### Exam Project for the course Federated Learning at the Leibniz University Hanover


## Abstract

This work investigates the applicability of Federated Learning (FL) for the continuous improvement of a neural network designed for pupil segmentation. This network is a core component of a portable device aimed at concussion detection through pupil dynamics analysis (refer to https://doi.org/10.1007/s11548-024-03128-9). Addressing the challenge of acquiring labeled medical images for training in a real-world deployment scenario, we explore a self-supervised approach using Monte Carlo Dropout inference within a Federated Learning framework. We evaluate and compare different Federated Learning strategies, specifically Federated Averaging and Federated Stochastic Gradient Descent, combined with variations in hyperparameters such as client data points per round and communication rounds. Our experiments, conducted using a U-Net architecture and synthetic infrared eye images, reveal that while Federated Learning with ground truth labels shows potential, our self-supervised MCD approach does not demonstrably outperform a well-pretrained centralized model in this specific use case. Furthermore, practical considerations concerning edge device training and the availability of labeled data for self-supervision in a medical product context lead us to conclude that Federated Learning may not be the most advantageous strategy for the continued improvement of our pupil segmentation model.

## Work

To get a deep insight on the domain and how we handled this task, refer to our paper FederatedLearningForPupilDynamics.pdf.

## Experiments

We tested approaches with Federated Averaging (FedAvg) and Fererated Stochastik Gradient Descent (FedSTD) with different sets of hyperparameters. Take a look at /code/*.ipynb to see the experiments setups and the results. The results are also stored in the /results directory.

## Installation

To run the Notebooks follow these steps:
1.  Clone this repository
2.  Navigate to the corresponding directory on your machine.
3.  Install python 3.12 and the required packages with pip3 install -r requirements.txt
4.  Make sure to run the experiment Notebooks from the /FederatedLearning directory
5.  Open Notebooks and run the cells
6.  Grab a cup of coffee and wait (depending on your graphics card)

