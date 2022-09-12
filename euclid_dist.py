import torch
import numpy as np


def euclid_dist(x1, x2):
    x_diff = x2[0] - x1[0]
    y_diff = x2[1] - x1[1]

    return torch.tensor([np.sqrt(x_diff ** 2 + y_diff ** 2)])
