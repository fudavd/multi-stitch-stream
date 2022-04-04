from . import *
from functools import partial
import cv2


def create_single_remap(x_maps: list = [], y_maps: list = []):
    remap_x = x_maps[0]
    remap_y = y_maps[0]
    for map_x, map_y in zip(x_maps[1:], y_maps[1:]):
        remap_x = cv2.remap(remap_x, map_x, map_y, cv2.INTER_NEAREST)
        remap_y = cv2.remap(remap_y, map_x, map_y, cv2.INTER_NEAREST)
    return remap_x, remap_y


def create_transform_function(map_x, map_y, roi=None):
    return partial(_create_transform_function, _map_y=map_y, _map_x=map_x, _roi=roi)


def _create_transform_function(img, _map_x, _map_y, _roi=None):
    img = cv2.remap(img, _map_x, _map_y, cv2.INTER_NEAREST)
    if _roi is not None:
        y, x, h, w = _roi
        img = img[y:y+h, x:x+w]
    return img
