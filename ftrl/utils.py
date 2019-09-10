import numpy as np


def shifted_scaled_sigmoid(x, shift=0, scale=1):
    s = 1 / (1 + np.exp(-x + shift))
    return s * scale
