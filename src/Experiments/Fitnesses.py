import numpy as np

from src.utils.Geometry import pix2meter, real_distance


def real_abs_dist(positions: np.array):
    relative_distance = positions[-1, :] - positions[0, :]
    return real_distance(relative_distance)

