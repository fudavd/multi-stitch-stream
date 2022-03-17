import getopt
import os
import sys
import time
from collections import deque
import cv2
import numpy as np


def geometric_median(X):
    summed_dist = []
    for row in X:
        summed_dist.append(np.linalg.norm(X - row).sum())
    return X[np.argmin(summed_dist)], np.argmin(summed_dist)


def load_barrel_map(file_path):
    data = np.load(file_path)
    return data['map_x'], data['map_y'], data['roi']


def correct_barrel_img(map_x, map_y, img):
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


class CalibrateBarrel:
    def __init__(self, id='cam1', output_dir='./Calibration_data/Barrel', debug=False, square_size=0.0013, load_settings=False):
        # args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])
        # args = dict(args)
        # args.setdefault('--debug', './output/')
        # args.setdefault('--square_size', '1')
        # args.setdefault('--threads', '4')

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

        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.n_squares, None)

        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

            cv2.drawChessboardCorners(img, self.n_squares, corners2, found)

            self.objpoints.append(self.pattern_points)
            self.imgpoints.append(corners)

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
                mean_error += error/len(self.objpoints)

            message = f"\nError:\n{mean_error}\n\ncamera matrix:\n{np.array(mtx).round()}\n\ndistortion coefficients:\n{dist_coefs.ravel()}"
            if len(self.objpoints) == self.objpoints.maxlen:
                self.remember(img, map_x, map_y, dist_coefs, roi)
                self.objpoints.clear()
                self.imgpoints.clear()
            else:
                # message = "Error: chessboard not found"
                pass

            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, line in enumerate(message.split("\n")):
                cv2.putText(img, line, (25, 25 * i + 25), font, 0.7, (0, 0, 255), 2)  # +- 10 for better display
            cv2.line(img, (int(w / 2) - 20, int(h / 2)), (int(w / 2) + 20, int(h / 2)), (0, 0, 0), 2)
            cv2.line(img, (int(w / 2), int(h / 2) - 20), (int(w / 2), int(h / 2) + 20), (0, 0, 0), 2)
        return img

    def remember(self, img, map_x, map_y, dc, roi):
        now = time.time()
        if self.prev_time + 1 > now:
            return
        else:
            self.dc.append(dc)
            self.Q.append((img, map_x, map_y, roi))
            # print("OK, ", np.std(self.dc, axis=0))
            print(self.Q.__len__(), self.Q.maxlen, np.all(np.std(self.dc, axis=0) / np.mean(self.dc, axis=0) < 1))
            if not np.all(np.std(self.dc, axis=0) / np.mean(self.dc, axis=0) < 1):
                self.dc.pop()
                self.Q.pop()
            if self.Q.__len__() == self.Q.maxlen and np.all(np.std(self.dc, axis=0) / np.mean(self.dc, axis=0) < 1):
                self.safe_calibration()
                try:
                    print("Saved calibration maps")
                    cal_param = load_barrel_map(os.path.join(self.output_dir, self.name + '.npz'))
                    self.map_x = cal_param[0]
                    self.map_y = cal_param[1]
                    self.roi = cal_param[2]
                except:
                    print("ERROR: Could not load calibration!")
            self.prev_time = now

    def safe_calibration(self):
        _, ind = geometric_median(np.array(self.dc).squeeze())
        img, map_x, map_y, roi = self.Q[ind]
        my_details = {
            'map_x': map_x,
            'map_y': map_y,
            'roi': roi
        }
        outfile = os.path.join(self.output_dir, self.name + '_chess.png')
        cv2.imwrite(outfile, img)
        np.savez(os.path.join(self.output_dir, self.name), **my_details)

