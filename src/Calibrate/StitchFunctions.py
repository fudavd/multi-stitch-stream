import copy
import os
import sys
from typing import List, Optional

import cv2
import numpy as np


# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M, mid_cut: bool = False):
    # Get width and height of input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = transform_corners(img2, M)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    tran_dist = [-x_min, -y_min]
    transform_array = translation_matrix(result_dims)

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img1, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))

    coord1 = img1_dims.squeeze()
    coord2 = img2_dims.squeeze()
    dist = np.array([np.linalg.norm(coord1[ind, :] - coord2, axis=1) for ind in range(4)])
    ind = np.unravel_index(np.argsort(dist, axis=None), dist.shape)
    # dir_vec = (coord1[ind[0][0]] - coord2[ind[1][0]]) / np.linalg.norm(coord1[ind[0][0]] - coord2[ind[1][0]])
    # offset = dir_vec * (coord1[ind[0][0]] - coord2[ind[1][0]]) / 2
    x_off = w1
    y_off = h1

    result_img[tran_dist[1]:y_off + tran_dist[1], tran_dist[0]:x_off + tran_dist[0]] = img2[:y_off, :x_off]

    map_transform_array = np.linalg.multi_dot((transform_array, M, np.array([[1, 0, 0],
                                                                             [0, 1, -h1],
                                                                             [0, 0, 1]])))
    map_x, map_y = cv2.initUndistortRectifyMap(np.eye(3), 0, None, map_transform_array,
                                               (x_max - x_min, y_max - y_min), 5)
    x_map, y_map = np.meshgrid(range(w1), range(h1))
    map_x[tran_dist[1]:y_off + tran_dist[1], tran_dist[0]:x_off + tran_dist[0]] = x_map[:y_off, :x_off]
    map_y[tran_dist[1]:y_off + tran_dist[1], tran_dist[0]:x_off + tran_dist[0]] = y_map[:y_off, :x_off]

    save_map = {
        'map_x': map_x,
        'map_y': map_y,
        'roi': (transform_array, M)
    }
    np.savez(os.path.join("cal_map"), **save_map)

    # Return the result
    return result_img, (x_off + tran_dist[0], y_off + tran_dist[1]), (tran_dist[0], tran_dist[1]), (
    tran_dist[0] + w1, tran_dist[1] + h1)


def transform_corners(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    img_dim = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(img_dim, H)


def translation_matrix(corners: np.ndarray) -> np.ndarray:
    offset = [min(0, min(corners[:, :, 0])), min(0, min(corners[:, :, 1]))]
    translation_array = np.array([[1, 0, -offset[0]],
                                  [0, 1, -offset[1]],
                                  [0, 0, 1]], dtype=np.float32)
    return translation_array


def compute_features(image, mask=None):
    """
    Compute the features and the keypoints of the image using SIFT.
    """
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, mask)


# Find SIFT and return Homography Matrix
def get_sift_homography(img1: np.array, img2: np.array, ratio=0.8, mask1=Optional[np.array], mask2=Optional[np.array]):
    """
    Compute HomographyMatrix between two images using SIFT.

    :param np.array img1: first image local frame
    :param np.array img2: second image global frame
    :param float ratio: Optional: ratio used for the Lowe's ratio test, Default = 0.8
    :param Optional[np.array] mask1: Optional: binary mask1 to look for features
    :param Optional[np.array] mask2: Optional: binary mask2 to look for features

    :return M: a tranformation matrix from local to global
    :returns keypoints: the positions of the relevant keypoints for (img1, img2)
    """
    # Extract keypoints and descriptors
    k1, d1 = compute_features(img1, mask1)
    k2, d2 = compute_features(img2, mask2)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)

    verified_matches = get_matches(d1, d2, ratio)

    # Mimnum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0, maxIters=2000)
        keypoints = (img1_pts.squeeze(), img2_pts.squeeze())
        return M, keypoints
    else:
        print('Error: Not enough matches')
        return -1


# Equalize Histogram of Color Images
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


def get_matches(descriptors_1: cv2.SIFT.descriptorType, descriptors_2: cv2.SIFT.descriptorType,
                ratio: float = 0.8) -> List:
    """
    Compute matches between descriptors_a and descriptors_b. Parameters

    :param cv2.SIFT.descriptorType descriptors_1: first image descriptors.
    :param cv2.SIFT.descriptorType descriptors_2: second image descriptors.
    :return List matches: Returns List of Matches between image_a and image_b.
    """

    matcher = cv2.DescriptorMatcher_create("BruteForce")
    full_match = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
    matches = []
    for m, n in full_match:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches
