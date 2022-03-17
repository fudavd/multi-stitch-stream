from typing import List

import numpy as np


def geometric_median(X):
    summed_dist = []
    for row in X:
        summed_dist.append(np.linalg.norm(X - row).sum())
    return X[np.argmin(summed_dist)], np.argmin(summed_dist)


def obtain_color_range(colors: List = None):
    s_min = 60
    v_min = 70

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([179, s_min, v_min])

    lower_white = np.array([0, 0, 185])
    upper_white = np.array([179, s_min, 255])

    lower_orange = np.array([20, s_min, v_min])
    upper_orange = np.array([30, 255, 255])

    lower_yellow = np.array([25, s_min, v_min])
    upper_yellow = np.array([35, 255, 255])

    lower_green = np.array([37, s_min, v_min])
    upper_green = np.array([80, 255, 255])

    lower_blue = np.array([85, s_min, v_min])
    upper_blue = np.array([140, 255, 255])

    lower_red1 = np.array([0, s_min, v_min])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([163, s_min, v_min])
    upper_red2 = np.array([179, 255, 255])

    color_range = {'green': (lower_green, upper_green),
                   'blue': (lower_blue, upper_blue),
                   'red1': (lower_red1, upper_red1),
                   'red2': (lower_red2, upper_red2),
                   'black': (lower_black, upper_black),
                   'white': (lower_white, upper_white),
                   'orange': (lower_orange, upper_orange),
                   'yellow': (lower_yellow, upper_yellow)}
    color_bounds_list = []
    for color in colors:
        if color == 'red':
            c_range = (color_range.get('red1'), color_range.get('red2'))
        else:
            c_range = color_range.get(color)
        color_bounds_list.append(c_range)
    assert color_bounds_list[0] is not None, f"Invalid color(s) provided: {colors}"
    return color_bounds_list
