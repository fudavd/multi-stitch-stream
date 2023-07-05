import numpy as np

from ..utils.Geometry import pix2meter, real_distance


def real_abs_dist(positions: np.ndarray):
    relative_distance = positions[-1, :] - positions[0, :]
    return real_distance(relative_distance[:2].reshape((1, 2))).squeeze()


def directed_locomotion(positions: np.ndarray, direction):
    relative_distance = positions[-1, :] - positions[0, :]
    dir_vector = np.array([np.cos(direction), np.sin(direction), 0])
    return np.matmul(dir_vector, relative_distance)


def left_step(positions: np.ndarray, orientations: np.ndarray):
    z_rot = np.unwrap(orientations - orientations[0, :])
    deviation = np.matmul(z_rot.T, z_rot) / len(z_rot)
    for_dist = directed_locomotion(positions, orientations[0, -1])
    side_dist = directed_locomotion(positions, orientations[0, -1] + 0.5 * np.pi)
    return side_dist - deviation - np.abs(for_dist)


def right_step(positions: np.ndarray, orientations: np.ndarray):
    z_rot = np.unwrap(orientations - orientations[0, :])
    deviation = np.matmul(z_rot.T, z_rot) / len(z_rot)
    for_dist = directed_locomotion(positions, orientations[0, -1])
    side_dist = directed_locomotion(positions, orientations[0, -1] - 0.5 * np.pi)
    return side_dist - deviation - np.abs(for_dist)


def unwrapped_rot(orientations: np.ndarray):
    rel_orientation = orientations - orientations[0, :]
    rot = np.unwrap(rel_orientation).squeeze()
    return rot[-1] - rot[0]


def signed_rot(orientation_vector: np.ndarray):
    total_angle = 0.0
    for i in range(1, len(orientation_vector)):
        u = orientation_vector[i - 1, :]
        v = orientation_vector[i, :]

        dot = u[0] * v[0] + u[1] * v[1]  # dot product between [x1, y1] and [x2, y2]
        det = u[0] * v[1] - u[1] * v[0]  # determinant
        delta = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

        total_angle += delta
    return total_angle
