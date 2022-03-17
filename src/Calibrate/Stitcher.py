import copy
from typing import List, Optional, Tuple
import os
import sys
from src.utils.utils import Image, PanoramaImage
import cv2
import numpy as np
import src.Calibrate.StitchFunctions as StitchFunctions
from src.utils.Filters import mask_colors
from src.utils.Measures import obtain_color_range


class PairStitcher:
    def __init__(self, images: List[Image], ratio=0.8,
                 mask_colors: Optional[List] = None,
                 matches: Optional[List] = None,
                 output_dir: str = './Calibration_data/Stitch/', verbose: bool = False, save_res: bool = False):
        """
        Create a new SingleImageStitcher object.

        :param List[Image] images: List of TWO images to compare
        :param float ratio: Optional: ratio used for the Lowe's ratio test, Default = 0.8
        :param Optional[List] mask_colors: Optional: list of standard colors to mask: ['green', 'blue', 'red', 'black', 'white', 'orange', yellow']
        :param Optional[List] matches: Optional: list of matches between both images
        :param str output_dir: output directory, Default = './Calibration_data/Stitch/'
        :param bool verbose: boolean, Default=False
        :param bool save_res: boolean, Default=False
        """
        self.output_dir = output_dir
        self.verbose = verbose
        self.save_res = save_res
        assert len(images) == 2, "PairStitcher can only handle 2 images"
        self.img1 = images[0]
        self.img2 = images[1]
        self.ratio = ratio
        self.colors = mask_colors
        self.k1, self.d1, self.k2, self.d2 = [None] * 4

        self.matches = matches
        self.H = None
        self.Hinv = None
        self.status = None
        self.overlap = None
        self.area_overlap = None
        self._I12 = None
        self._I21 = None
        self.matchpts1 = None
        self.matchpts2 = None
        self.mask = [None, None]

    def isin(self, image: Image) -> bool:
        """
        Check if stitcher contains given image

        :param Image image: image to check
        :return bool: boolean
        """
        return self.img1 == image or self.img2 == image

    def key_features(self, mask1=None, mask2=None):
        self.k1, self.d1 = StitchFunctions.compute_features(self.img1.image, mask1)
        self.k2, self.d2 = StitchFunctions.compute_features(self.img2.image, mask2)
        self.img1.features = self.d1
        self.img2.features = self.d2
        self.img1.keypoints = self.k1
        self.img2.keypoints = self.k2

    def find_match_points(self):
        self.matches = StitchFunctions.get_matches(self.d1, self.d2, self.ratio)
        self.matchpts1 = np.float32([self.img1.keypoints[match.queryIdx].pt for match in self.matches])
        self.matchpts2 = np.float32([self.img2.keypoints[match.trainIdx].pt for match in self.matches])

    def sift_homography(self, ransac_reproj_thresh: float = 5.0, ransac_max_iter: int = 2000):
        """
        Compute HomographyMatrix between two images using SIFT.

        :param float ransac_reproj_thresh: threshold used for RANSAC, Default = 5
        :param int ransac_max_iter: maximum number of iterations to estimate HomographyMatrix, Default = 2000
        """
        if self.colors is not None:
            colors = obtain_color_range(self.colors)
            for ind, img in enumerate([self.img1.image, self.img2.image]):
                img_blur = cv2.medianBlur(copy.deepcopy(img), 3)
                hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
                mask = mask_colors(hsv, colors)
                # binary image
                _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
                self.mask[ind] = binary
        self.key_features(self.mask[0], self.mask[1])
        self.find_match_points()
        matchpts = [self.matchpts1, self.matchpts2]

        self.H, self.status = cv2.findHomography(
            matchpts[1],
            matchpts[0],
            cv2.RANSAC,
            ransac_reproj_thresh,
            maxIters=ransac_max_iter)

        self.Hinv, _ = cv2.findHomography(
            matchpts[0],
            matchpts[1],
            cv2.RANSAC,
            ransac_reproj_thresh,
            maxIters=ransac_max_iter)

    def set_overlap(self) -> None:
        """
        Compute and set the overlap region between the two images.
        """
        if self.H is None:
            self.sift_homography()

        mask_a = np.ones_like(self.img1.image[:, :, 0], dtype=np.uint8)
        mask_b = cv2.warpPerspective(np.ones_like(self.img2.image[:, :, 0], dtype=np.uint8),
                                     self.H,
                                     mask_a.shape[::-1]
                                     )

        self.overlap = mask_a * mask_b
        self.area_overlap = self.overlap.sum()

    def stitch(self, cov_min: float = 8, cov_rat: float = 0.3, verbose: bool = False) -> bool:
        """
        Stitch image pairs

        :param cov_min: minimum number of key-points within overlap, Default=8
        :param cov_rat: minimum ratio of key-points in overlap, Default=0.3
        :param verbose: verbose, Default=False
        :return: Boolean, stict validity: if cov_min and cov_rat are met
        """
        if self.overlap is None:
            self.set_overlap()

        if self.status is None:
            self.sift_homography()

        matches_in_overlap = self.matchpts1[self.overlap[self.matchpts1[:, 1].astype(np.int64),
                                                         self.matchpts1[:, 0].astype(np.int64)] == 1]
        min_inliers_match = np.ceil(cov_min + cov_rat * matches_in_overlap.shape[0])
        valid_match = self.status.sum() > min_inliers_match
        if valid_match:
            if self.verbose or verbose:
                result_image, off_set, tran_dist1, tran_dist2 = StitchFunctions.get_stitched_image(self.img2.image,
                                                                                                   self.img1.image,
                                                                                                   self.H)
                for (im1_pt, im2_pt) in zip(self.matchpts1, self.matchpts2):
                    img1 = cv2.drawMarker(self.img1.image, im1_pt.astype(int),
                                          [0, 0, 256], markerType=3, markerSize=10, thickness=2)
                    img2 = cv2.drawMarker(self.img2.image, im2_pt.astype(int),
                                          [0, 0, 256], markerType=3, markerSize=10, thickness=2)
                img_show1 = cv2.resize(img1, (720, 405))
                img_show2 = cv2.resize(img2, (720, 405))
                input_images = np.hstack((img_show1, img_show2))
                if self.colors is not None:
                    img_mask1 = cv2.resize(self.mask[0], (720, 405))
                    img_mask2 = cv2.resize(self.mask[1], (720, 405))
                    input_images = np.vstack((input_images, np.repeat(np.expand_dims(np.hstack((img_mask1, img_mask2)),
                                                                                     axis=2), 3, axis=2)))
                cv2.imshow('Input Images', input_images)
                cv2.waitKey()
                # Show the resulting image
                result_image = cv2.line(result_image, (off_set[0], tran_dist1[1]), off_set, color=(0, 0, 255), thickness=2)
                result_image = cv2.line(result_image, (tran_dist1[0], off_set[1]), off_set,  color=(0, 0, 255), thickness=2)
                result_image = cv2.line(result_image, (tran_dist2[0], tran_dist1[1]), tran_dist2, color=(255, 125, 0), thickness=2)
                result_image = cv2.line(result_image, (tran_dist1[0  ], tran_dist2[1]), tran_dist2, color=(255, 125, 0), thickness=2)
                cv2.imshow('Result',
                           cv2.resize(result_image, (int(result_image.shape[1] / 2), int(result_image.shape[0] / 2))))
                cv2.waitKey()
                cv2.destroyAllWindows()
                if self.save_res:
                    # Write the result to the same directory
                    cv2.imwrite(self.output_dir + f'stitch:{self.img1.name, self.img2.name}.png', result_image)
        else:
            if self.verbose or verbose:
                print(f"Not enough overlap found!!! {self.status.sum()} \t{round(min_inliers_match,2)}\t matchpoints in overlap required minimum")
        return valid_match

    @property
    def I12(self):
        if self._I12 is None:
            self.set_intensities()
        return self._I12

    @I12.setter
    def I12(self, I12):
        self._I12 = I12

    @property
    def I21(self):
        if self._I21 is None:
            self.set_intensities()
        return self._I21

    @I21.setter
    def I21(self, I21):
        self._I21 = I21

    def set_intensities(self) -> None:
        """
        Set intensities of the two images in the overlap region.
        """
        if self.overlap is None:
            self.set_overlap()

        inverse_overlap = cv2.warpPerspective(
            self.overlap, self.Hinv, self.img2.image.shape[1::-1]
        )

        if self.overlap.sum() == 0:
            print(self.img1.name, self.img2.name)

        self._I12 = (
                np.sum(
                    self.img1.image * np.repeat(self.overlap[:, :, np.newaxis], 3, axis=2),
                    axis=(0, 1),
                )
                / self.overlap.sum()
        )
        self._I21 = (
                np.sum(
                    self.img2.image * np.repeat(inverse_overlap[:, :, np.newaxis], 3, axis=2),
                    axis=(0, 1),
                )
                / inverse_overlap.sum()
        )


class MultiStitcher:
    def __init__(self, images: List[Image], ratio=0.3,
                 mask_colors: Optional[List] = None,
                 output_dir: str = f'./Calibration_data/Stitch/', verbose: bool = False):
        """
        Create a new MultiImageMatches object.

        :param List[Image] images: List of images to compare.
        :param float ratio: Optional: ratio used for the Lowe's ratio test, by default = 0.8.
        :param Optional[List] mask_colors: List of standard colors to mask ['green', 'blue', 'red1', 'black', 'white', 'orange', yellow']
        :param str output_dir: output directory, Default = './Calibration_data/Stitch/'
        :param bool verbose: boolean, Default=False
        """
        self.output_dir = output_dir
        self.verbose = verbose

        self.images = images
        self.matches = {image.name: {} for image in images}
        self.ratio = ratio
        self.colors = mask_colors
        self.panorama = PanoramaImage("pan_test", self.images)

        self.paired_matches = None
        self.connected_components = None

    def get_matches(self, img1: Image, img2: Image) -> List:
        """
        Obtain matches for the given images and place them in the dictionary

        :param Image img1: First image.
        :param Image img2: Second image.
        :return: List of matches between the two images.
        """
        if img1.features is None:
            img1.compute_features()
        if img2.features is None:
            img2.compute_features()
        if img2.name not in self.matches[img1.name]:
            matches = StitchFunctions.get_matches(img1.features, img2.features, self.ratio)
            self.matches[img1.name][img2.name] = matches

        return self.matches[img1.name][img2.name]

    def get_pair_matches(self, max_images: Optional[int] = 6):
        """
        Get the pair matches for the given images.

        :param Optional[int] max_images: Optional: maximum number of matches for each image, Default=6
        :returns List[PairStitcher]: List of pair stitched pairs.
        """
        pair_matches = []
        for i, img1 in enumerate(self.images):
            possible_matches = sorted(
                self.images[:i] + self.images[i + 1:],
                key=lambda image, ref=img1: len(self.get_matches(ref, image)),
                reverse=True,
            )[:max_images]
            for img2 in possible_matches:
                if self.images.index(img2) > i:
                    pair_match = PairStitcher([img1, img2], matches=self.get_matches(img1, img2), verbose=self.verbose)
                    if pair_match.stitch():
                        pair_matches.append(pair_match)
        self.paired_matches = pair_matches

    def compute_connected_components(self):
        """
        Compute and save a List of connected Images for the given pair matches
        """
        connected_components = []
        pair_matches_to_check = self.paired_matches.copy()
        component_id = 0
        while len(pair_matches_to_check) > 0:
            pair_match = pair_matches_to_check.pop(0)
            connected_component = {pair_match.img1, pair_match.img2}
            size = len(connected_component)
            stable = False
            while not stable:
                i = 0
                while i < len(pair_matches_to_check):
                    pair_match = pair_matches_to_check[i]
                    if (
                            pair_match.img1 in connected_component
                            or pair_match.img2 in connected_component
                    ):
                        connected_component.add(pair_match.img1)
                        connected_component.add(pair_match.img2)
                        pair_matches_to_check.pop(i)
                    else:
                        i += 1
                stable = size == len(connected_component)
                size = len(connected_component)
            connected_components.append(list(connected_component))
            for image in connected_component:
                image.component_id = component_id
            component_id += 1
        self.connected_components = connected_components

    def build_homographies(self) -> None:
        """
        Build homographies for each image of each connected component, using the pair matches.
        The homographies are saved in the images themselves.
        """
        pair_matches: List[PairStitcher] = self.paired_matches
        for connected_component in self.connected_components:
            component_matches = [
                pair_match for pair_match in pair_matches if pair_match.img1 in connected_component
            ]

            images_added = set()
            current_homography = np.eye(3)

            pair_match = component_matches[0]
            pair_match.sift_homography()

            nb_pairs = len(pair_matches)

            if sum([10 * (nb_pairs - i) for i, match in enumerate(pair_matches) if match.isin(pair_match.img1)]) > \
               sum([10 * (nb_pairs - i) for i, match in enumerate(pair_matches) if match.isin(pair_match.img2)]):
                pair_match.img1.H = np.eye(3)
                pair_match.img2.H = pair_match.H
            else:
                pair_match.img2.H = np.eye(3)
                pair_match.img1.H = pair_match.Hinv

            images_added.add(pair_match.img1)
            images_added.add(pair_match.img2)

            while len(images_added) < len(connected_component):
                for pair_match in component_matches:
                    if pair_match.img1 in images_added and pair_match.img2 not in images_added:
                        pair_match.sift_homography()
                        homography = pair_match.H @ current_homography
                        pair_match.img2.H = pair_match.img1.H @ homography
                        images_added.add(pair_match.img2)
                        break

                    elif pair_match.img1 not in images_added and pair_match.img2 in images_added:
                        pair_match.sift_homography()
                        homography = pair_match.Hinv @ current_homography
                        pair_match.img1.H = pair_match.img2.H @ homography
                        images_added.add(pair_match.img1)
                        break

    def stitch(self, name: str="panorama", type: str="pareto", save_res: bool=False) -> np.array:
        """
        Create panorama of stitched images

        :param name: panorma image name (only important when savinf result), Default=panorama
        :param type: type of stitching, Default=pareto
        :param save_res: if true save results in self.output_dir
        :return: panorama image
        """
        if self.verbose:
            print("MultiStitcher: find matching pairs")
        if self.connected_components is None:
            self.get_pair_matches()
            self.paired_matches.sort(key=lambda pair_match: len(self.matches), reverse=True)
            self.compute_connected_components()
        if self.verbose:
            print("MultiStitcher: create stitch-pair homographies")
        self.build_homographies()

        if self.verbose:
            print("MultiStitcher: stitch panorama")
        for connected_component in self.connected_components:
            self.panorama = PanoramaImage(name, connected_component)
            self.panorama.calculate_global_h()
            self.panorama.stitch_panorama(type)

        if save_res:
            if self.verbose:
                print(f"Saving results to {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
            self.panorama.save_image(self.output_dir)

        return self.panorama.panorama