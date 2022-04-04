import copy
import os
from typing import Tuple, List
import cv2
import numpy as np

from src.Calibrate import StitchFunctions


def search_file_list(rootname, file_name):
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    return file_list


def bernoulli(p, shape):
    return np.random.binomial(1, p, shape)


def nearest(data, reference):
    val = min(data, key=lambda x: abs(x - reference))
    index = np.argmin(abs(np.array(data) - val))
    return val, index


class Image:
    def __init__(self, name: str, path: str = None, size: int = None, out_dir: str = './Calibration_data/Stitch/'):
        """
        Image object

        :param str name: Image name used to identify
        :param str path: path to image
        :param int size: maximum Image size when resizing
        :param str out_dir: output directory to save image
        """
        self.name = name
        self.output_dir = out_dir
        self.path = path
        self.image = None
        if path is not None:
            self.image: np.ndarray = cv2.imread(self.path)
        if self.image is not None:
            if size is not None:
                h, w = self.image.shape[:2]
                if max(w, h) > size:
                    if w > h:
                        self.image = cv2.resize(self.image, (size, int(h * size / w)))
                    else:
                        self.image = cv2.resize(self.image, (int(w * size / h), size))
            self.h, self.w = self.image.shape[:2]
        self.keypoints = None
        self.features = None
        self.H: np.ndarray = np.eye(3)
        self.Hinv: np.ndarray = np.eye(3)
        self.component_id: int = 0
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)

    def compute_features(self):
        """
        Compute the features and the keypoints of the image using SIFT.
        """
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(self.image, None)
        self.keypoints = keypoints
        self.features = features

    def build_weight_matrix(self, type: str = "radial") -> np.ndarray:
        """
        Create a 2D weights matrix (h x w) with type shape Isoline's. Center weight = 1.0; Edge weight = 0.0001

        :param str type: shape of Isoline [radial, square, cross, diamond], Default = radial
        :return: np.Array with size self.image.shape
        """
        h_vec = np.arange(self.h)
        h_vec = np.minimum(h_vec, h_vec[::-1])
        h_vec = h_vec / np.max(h_vec)
        w_vec = np.arange(self.w)
        w_vec = np.minimum(w_vec, w_vec[::-1])
        w_vec = w_vec / np.max(w_vec)
        if type == "radial":
            weight_matrix = np.outer(h_vec, w_vec)
        if type == "square":
            weight_matrix = np.minimum.outer(h_vec, w_vec)
        if type == "cross":
            weight_matrix = np.maximum.outer(h_vec, w_vec)
        if type == "diamond":
            weight_matrix = np.add.outer(h_vec, w_vec) / 2
        return weight_matrix

    def save_image(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        cv2.imwrite(output_dir + f'{self.name}.png', self.image)


class PanoramaImage(Image):
    def __init__(self, name: str, images: List[Image]):
        super().__init__(name)
        self.panorama = self.image
        self.weights = None
        self.offset = np.eye(3)
        self.w = 0
        self.h = 0
        self.images = images
        self.weights = []
        self.H_global = []
        self.map_x = None
        self.map_y = None

    def stitch_panorama(self, type: str = "linear_blend", weight_ptrn: str = "radial"):
        """
        Stitch the panorama according with their current Homography
        :param type: Type of stitching, Default = 'linear_blend'
        :param weight_ptrn: Type of weight pattern used for image stitching, Default = 'radial'
        :return:
        """
        self.panorama = np.zeros((self.h, self.w, 3)).astype(dtype=np.uint8)
        self.weights = []
        for ind, image in enumerate(self.images):
            self.weights.append(
                cv2.warpPerspective(image.build_weight_matrix(weight_ptrn), self.H_global[ind], (self.w, self.h)))
        weight_sum = np.sum(self.weights, axis=0)
        self.weights = np.divide(self.weights, weight_sum, where=weight_sum != 0)

        if type == "linear_blend":
            for ind, image in enumerate(self.images):
                cur_im = cv2.warpPerspective(image.image, self.H_global[ind], (self.w, self.h))
                self.panorama += np.where(np.repeat(np.sum(cur_im, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
                                          0,
                                          cur_im * np.dstack([self.weights[ind] for _ in range(3)])).astype(np.uint8)
        if type == "pareto":
            max_weight_ind = np.where(weight_sum == 0,
                                      -1,
                                      np.argmax(self.weights, axis=0))
            x_map = np.zeros((self.h, self.w), dtype=np.float32)
            y_map = np.zeros((self.h, self.w), dtype=np.float32)
            image_stack = []
            for ind, image in enumerate(self.images):
                stack_translation = np.array([[1, 0, 0],
                                              [0, 1, -image.h * ind],
                                              [0, 0, 1]], dtype=np.float32)
                map_transform_array = self.H_global[ind] @ stack_translation
                map_x, map_y = cv2.initUndistortRectifyMap(np.eye(3), 0, None, map_transform_array,
                                                           (self.w, self.h), 5)
                x_map += np.where(max_weight_ind != ind, -1.0, map_x)
                y_map += np.where(max_weight_ind != ind, -1.0, map_y)
                image_stack.append(image.image)
            self.panorama = cv2.remap(np.vstack(image_stack), np.floor(x_map), np.floor(y_map),
                                      cv2.INTER_NEAREST,
                                      cv2.BORDER_ISOLATED)
            self.map_x = x_map
            self.map_y = y_map
            maps = {
                'map_x': x_map,
                'map_y': y_map,
            }
            np.savez(os.path.join(self.output_dir, self.name), **maps)
        self.image = self.panorama

    def calculate_global_h(self):
        """
        build homographies for sequential stitching of all image pairs in the global panorama frame of reference

        :return: List of global homographies ordered with respect to self.images
        """
        for image in self.images:
            H = self.offset @ image.H

            corners = StitchFunctions.transform_corners(image.image, H)
            added_offset = StitchFunctions.translation_matrix(corners)

            corners_image = StitchFunctions.transform_corners(image.image, added_offset @ H)
            corners_image = np.ceil(corners_image).astype(int)

            w_img, h_img = np.max(corners_image, axis=(1, 0))

            self.w, self.h, _ = np.ceil(added_offset @ np.array([self.w, self.h, 1])).astype(int)
            self.w = np.max((self.w, w_img))
            self.h = np.max((self.h, h_img))

            self.H_global.append(H)
            for ind, _H in enumerate(self.H_global):
                self.H_global[ind] = added_offset @ _H
            self.offset = added_offset @ self.offset
        return self.H_global

    def set_global_h(self, h_glob: List[np.array]):
        """
        Set glabal homographies
        :param h_glob: List of global homographies ordered with respect to self.images
        :return:
        """
        self.H_global = h_glob
        return

    def get_global_h(self):
        """
        Get current list of global homographies
        :return: List of global homographies ordered with respect to self.images
        """
        return self.H_global
