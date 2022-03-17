import getopt
import os
import sys
import time
from collections import deque
import cv2
import numpy as np
import cv2.aruco as aruco

from ..utils.Geometry import obtain_image_hull
from ..utils.Measures import geometric_median


def generate_charuco(path: str = "./Calibration_data/Barrel"):
    # Charuco involves markers from this dictionary
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    # Length of one side of square in the chessboard
    square = 0.04
    # Length ofone side of markers located in white squares of chessboard
    marker = 0.02
    # Watch for column and row number of chessboard
    board = aruco.CharucoBoard_create(5, 7, square, marker, marker_dict)
    # Pixel number for every square
    charuco_img = board.draw((200 * 5, 200 * 7))
    # Name of the file
    filename_str = "CHARUCO_DICT_5_7_{}_{}_6X6_250.jpg".format(square, marker)
    cv2.imwrite(path + filename_str, charuco_img)


def load_barrel_map(file_path):
    data = np.load(file_path)
    return data['map_x'], data['map_y'], data['roi']


def correct_barrel_img(map_x, map_y, img):
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


class CalibrateBarrelAruco:
    def __init__(self, id='cam1', output_dir='./Calibration_data/Barrel', debug=False, square_size=0.0013,
                 load_settings=False):
        CHARUCOBOARD_ROWCOUNT = 7
        CHARUCOBOARD_COLCOUNT = 5
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=CHARUCOBOARD_COLCOUNT,
            squaresY=CHARUCOBOARD_ROWCOUNT,
            squareLength=0.04,
            markerLength=0.02,
            dictionary=self.ARUCO_DICT)

        self.output_dir = output_dir
        self.name = id

        self.debug_dir = os.path.join(output_dir, 'debug')
        if debug:
            if self.debug_dir and not os.path.isdir(self.debug_dir):
                os.mkdir(self.debug_dir)
            self.debug_dir = self.debug_dir

        self.n_squares = (CHARUCOBOARD_ROWCOUNT, CHARUCOBOARD_COLCOUNT)
        pattern_points = np.zeros((np.prod(self.n_squares), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.n_squares).T.reshape(-1, 2)
        pattern_points *= square_size

        self.pattern_points = np.array([pattern_points])

        buffer_size = 100
        self.img_buffer = deque(maxlen=buffer_size)
        self.corners_buffer = deque(maxlen=buffer_size)
        self.img_id_buffer = deque(maxlen=buffer_size)

        self.coverage = None
        self.prev_time = time.time()

        self.map_y = None
        self.map_x = None
        self.roi = None
        if load_settings:
            cal_param = load_barrel_map(os.path.join(self.output_dir, self.name + '.npz'))
            self.map_x = cal_param[0]
            self.map_y = cal_param[1]
            self.roi = cal_param[2]

    def process_image(self, img):
        if np.any(self.map_x):
            img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
            x, y, w, h = self.roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.putText(img, 'Cal: y/n', (x + 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 25, 0), 2)
            return img

        if self.coverage is None:
            self.coverage = np.zeros_like(img[::0])

        img_save = np.asarray(img, dtype="int32")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.n_squares, None)
        if found:
            # Find aruco markers in the query image
            corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=self.ARUCO_DICT)

            # Outline the aruco markers found in our query image
            img = aruco.drawDetectedMarkers(image=img, corners=corners)

            # Get charuco corners_buffer and ids from detected aruco markers
            response, char_corners, char_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=self.CHARUCO_BOARD)

            if response > 20:

                # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                img = aruco.drawDetectedCornersCharuco(image=img, charucoCorners=char_corners, charucoIds=char_ids)
                self.remember(img_save, char_corners, char_ids)

            else:
                # message = "Error: chessboard not found"
                pass

            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, line in enumerate(message.split("\n")):
                cv2.putText(img, line, (25, 25 * i + 25), font, 0.7, (0, 0, 255), 2)  # +- 10 for better display
            cv2.line(img, (int(w / 2) - 20, int(h / 2)), (int(w / 2) + 20, int(h / 2)), (0, 0, 0), 2)
            cv2.line(img, (int(w / 2), int(h / 2) - 20), (int(w / 2), int(h / 2) + 20), (0, 0, 0), 2)
        return img

    def remember(self, img, corners, im_ids):
        now = time.time()
        if self.prev_time + 1 > now:
            return
        else:
            self.img_buffer.append(img)
            self.corners_buffer.append(corners)
            self.img_id_buffer.append(im_ids)

            # print("OK, ", np.std(self.dc, axis=0))
            print(self.img_buffer.__len__(), self.img_buffer.maxlen, self.coverage)
            if not np.all(np.std(self.img_buffer, axis=0) / np.mean(self.img_buffer, axis=0) < 1):
                self.img_buffer.pop()
                self.corners_buffer.pop()
                self.img_id_buffer.pop()
            if self.img_buffer.__len__() == self.img_buffer.maxlen and self.coverage > 0.8:
                self.safe_calibration(img.shape[:2])
                try:
                    print("Saved calibration maps")
                    cal_param = load_barrel_map(os.path.join(self.output_dir, self.name + '.npz'))
                    self.map_x = cal_param[0]
                    self.map_y = cal_param[1]
                    self.roi = cal_param[2]
                except:
                    print("ERROR: Could not load calibration!")
            self.prev_time = now

    def safe_calibration(self, img_size):
        # Make sure at least one image was found
        if len(self.img_id_buffer) < self.img_buffer.maxlen:
            # Calibration failed because there were no images, warn the user
            print(f"Missing {len(self.img_id_buffer)}/{self.img_buffer.maxlen}")

        # Now that we've seen all of our images, perform the camera calibration
        # based on the set of points we've discovered
        calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=self.corners_buffer,
            charucoIds=self.img_id_buffer,
            board=self.CHARUCO_BOARD,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None)

        # Print matrix and distortion coefficient to the console
        print(cameraMatrix)
        print(distCoeffs)

        cv_file = cv2.FileStorage(os.path.join(self.output_dir, self.name + "cameraParameters.xml"),
                                  cv2.FILE_STORAGE_WRITE)
        cv_file.write(os.path.join(self.output_dir, self.name + "cameraMatrix"), cameraMatrix)
        cv_file.write(os.path.join(self.output_dir, self.name + "dist_coeffs"), distCoeffs)

        # note you *release* you don't close() a FileStorage object
        cv_file.release()


class CalibrateBarrel:
    def __init__(self, id='cam1', output_dir='./Calibration_data/Barrel', debug=False, square_size=0.0013,
                 load_settings=False):
        self.output_dir = output_dir
        self.name = id

        self.debug_dir = os.path.join(output_dir, 'debug')
        if debug:
            if self.debug_dir and not os.path.isdir(self.debug_dir):
                os.mkdir(self.debug_dir)
            self.debug_dir = self.debug_dir

        self.n_squares = (9, 9)
        pattern_points = np.loadtxt('Calibration_data/Real/checker_board.csv')

        self.pattern_points = np.array([pattern_points])
        self.objpoints = deque(maxlen=25)
        self.imgpoints = deque(maxlen=25)
        self.covered = None
        self.cov_rad = None

        self.Q = deque(maxlen=15)
        self.prev_time = time.time()

        self.dc = deque(maxlen=15)

        self.map_y = None
        self.map_x = None
        self.roi = None
        if load_settings:
            cal_param = load_barrel_map(os.path.join(self.output_dir, self.name + '.npz'))
            self.map_x = cal_param[0]
            self.map_y = cal_param[1]
            self.roi = cal_param[2]

    def process_image(self, img):
        if np.any(self.map_x):
            img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
            x, y, w, h = self.roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.putText(img, 'Cal: y/n', (x + 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 25, 0), 2)
            return img

        if self.covered is None:
            self.covered = np.ones_like(img) * 255
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.n_squares, None)

        if found:
            if self.cov_rad is None:
                hull = obtain_image_hull(img, corners)
                self.cov_rad = int(hull[1].area/10)
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

            cv2.drawChessboardCorners(img, self.n_squares, corners2, found)

            covered = 0
            chess_centroid = corners.mean(axis=0)
            for points in self.imgpoints:
                distance = np.mean(points, axis=0) - chess_centroid
                covered += np.hypot(distance[:, 0], distance[:, 1]) < self.cov_rad
            if covered <= 3:
                self.objpoints.append(self.pattern_points)
                self.imgpoints.append(corners)
                self.covered = cv2.circle(self.covered, chess_centroid.astype(int).squeeze(),
                                       self.cov_rad, (-255, -255, -255), -1)
            else:
                print("Area already covered")
        img = cv2.bitwise_and(img, self.covered)
        if self.imgpoints.__len__() == self.imgpoints.maxlen:
            img = self.calibrate_images(img, h, w)
        cv2.line(img, (int(w / 2) - 20, int(h / 2)), (int(w / 2) + 20, int(h / 2)), (0, 0, 0), 2)
        cv2.line(img, (int(w / 2), int(h / 2) - 20), (int(w / 2), int(h / 2) + 20), (0, 0, 0), 2)
        return img

    def calibrate_images(self, img, h, w):
        # calculate camera distortion
        rms, mtx, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (w, h), None, None)
        # Generate new camera matrix from parameters
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coefs, (w, h), 1, (w, h))
        # Generate LUTs for undistortion
        map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist_coefs, None, new_mtx, (w, h), 5)

        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist_coefs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error / len(self.objpoints)
        message = f"\nError:\n{mean_error}\n\ncamera matrix:\n{np.array(mtx).round()}\n\ndistortion coefficients:\n{dist_coefs.ravel()}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(message.split("\n")):
            cv2.putText(img, line, (25, 25 * i + 25), font, 0.7, (0, 0, 255), 2)  # +- 10 for better display
        my_details = {
            'map_x': map_x,
            'map_y': map_y,
            'roi': roi,
            'H': (mtx, new_mtx)
        }
        outfile = os.path.join(self.output_dir, self.name + '_chess.png')
        cv2.imwrite(outfile, img)
        np.savez(os.path.join(self.output_dir, self.name), **my_details)
        return img

