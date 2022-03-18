import traceback
import sys
import time
from typing import List

import cv2
import numpy as np

from src.utils.Geometry import pix2meter
from src.utils.Filters import mask_colors
from src.utils.Measures import obtain_color_range


class MotionCapture:
    def __init__(self, colors=None,):
        if colors is None:
            colors = ['green', 'blue', 'red']

        # range_list = obtain_color_range(self.colors)
        self.total_area_num_dict = {}
        for color in colors:
            self.total_area_num_dict[color] = (30, obtain_color_range([color]))
        self.tolerance = 5
        self.pix2real = pix2meter

    def get_robot_pos(self, img):
        Img = img.copy()
        contours_area = {}
        hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
        for target_color, (area, c_range) in self.total_area_num_dict.items():
            hsv = cv2.medianBlur(hsv, 5)
            mask = mask_colors(hsv, c_range)
            # dilation = cv2.GaussianBlur(mask, (3, 3), 0)

            # binary image
            ret, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

            # search contours and rank them by the size of areas
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[::-1]
            contours_area[target_color] = contours[:area]

        robot_label = []
        robot_contour = []
        robot_data = []

        for b_color, contours_big in contours_area.items():
            for contour_big in contours_big:
                x_big, y_big, w_big, h_big = cv2.boundingRect(contour_big)
                # epsilon = 0.1 * cv2.arcLength(contour_big, True)
                # approx = cv2.approxPolyDP(contour_big, epsilon, True)
                real_w_big = self.pix2real(w_big)
                real_h_big = self.pix2real(h_big)
                if not ((0.05 < real_h_big < 0.5) and (0.05 < real_w_big < 0.5)):
                    continue

                for s_color, contours_small in contours_area.items():
                    if b_color == s_color:
                        continue

                    for contour_small in contours_small:
                        x_small, y_small, w_small, h_small = cv2.boundingRect(contour_small)
                        real_w_small = self.pix2real(w_small)
                        real_h_small = self.pix2real(h_small)
                        distance = np.hypot(x_small - x_big, y_small - y_big)
                        if not (0.05 < self.pix2real(distance) < 0.25) or \
                                w_small * h_small > w_big * h_big or \
                                not ((0.03 < real_w_small < 0.25) and (0.03 < real_h_small < 0.25)):
                            continue
                        direction_vector_x = (x_small + w_small / 2) - (x_big + w_big / 2)
                        direction_vector_y = (y_small + h_small / 2) - (y_big + h_big / 2)
                        robot_label += [b_color[0] + s_color[0]]
                        robot_contour += [(x_big, y_big, w_big, h_big, x_small, y_small, w_small, h_small)]
                        norm = np.sqrt(direction_vector_x ** 2 + direction_vector_y ** 2)
                        direction_vector_x /= norm
                        direction_vector_y /= norm
                        robot_data += [
                            (x_big + w_big / 2, (y_big + h_big / 2), direction_vector_x, direction_vector_y)]

        for idx, rect in enumerate(robot_contour):
            x, y, w, h, x2, y2, w2, h2 = rect
            cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255,), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(Img, f"{robot_label[idx]} | size: {w * h}, {w2 * h2}", (x + 14, y + 14), font, 0.7, (0, 0, 255),
                        2)  # +- 10 for better display
            lineThickness = 2
            cv2.line(Img, (int(robot_data[idx][0]), int(robot_data[idx][1])),
                     (int(robot_data[idx][0] + robot_data[idx][2] * 30),
                      int(robot_data[idx][1] + robot_data[idx][3] * 30)),
                     (0, 255, 0), lineThickness)
        return Img


class MotionCaptureRobot:
    def __init__(self, robot_id: str, colors: List[str], return_img=False):
        self.return_img = return_img
        self.robot_id = robot_id
        self.colors = colors
        self.contour_colors = (obtain_color_range([colors[0]]), obtain_color_range([colors[1]]))
        self.tolerance = 5
        self.pix2real = pix2meter
        self.robot_states = np.empty(4)
        self.t = []
        self.prev_state = None

    def log_robot_pos(self, img):
        Img = img.copy()
        Img_r = cv2.resize(Img.copy(), (int(Img.shape[1]/2), int(Img.shape[0]/2)))
        contours_area = []
        Img_r = cv2.medianBlur(Img_r, 3)
        hsv = cv2.cvtColor(Img_r, cv2.COLOR_BGR2HSV)
        for c_range in self.contour_colors:
            mask = mask_colors(hsv, c_range)
            mask = cv2.GaussianBlur(mask,  (5, 5), 0)
            contours_r = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            contours = []
            for contour in contours_r:
                x_big, y_big, w_big, h_big = cv2.boundingRect(contour)
                real_w_big = self.pix2real(w_big)
                real_h_big = self.pix2real(h_big)
                if not ((0.015 < real_h_big < 0.25) and (0.015 < real_w_big < 0.25)):
                    continue
                contour[:, :, 0] = contour[:, :, 0] * 2
                contour[:, :, 1] = contour[:, :, 1] * 2
                contours.append(contour)
            contours_area.append(contours)

        robot_contour = []
        robot_data = []
        if len(contours_area) == 2:
            for contour_big in contours_area[0]:
                x_big, y_big, w_big, h_big = cv2.boundingRect(contour_big)
                for contour_small in contours_area[1]:
                    x_small, y_small, w_small, h_small = cv2.boundingRect(contour_small)
                    real_w_small = self.pix2real(w_small)
                    real_h_small = self.pix2real(h_small)
                    distance = np.hypot(x_small - x_big, y_small - y_big)
                    if not (0.05 < self.pix2real(distance) < 0.1) or \
                            w_small * h_small > w_big * h_big or \
                            not ((0.03 < real_w_small < 0.25) and (0.03 < real_h_small < 0.25)):
                        continue
                    direction_vector_x = (x_small + w_small / 2) - (x_big + w_big / 2)
                    direction_vector_y = (y_small + h_small / 2) - (y_big + h_big / 2)
                    robot_contour += [(x_big, y_big, w_big, h_big, x_small, y_small, w_small, h_small)]
                    norm = np.sqrt(direction_vector_x ** 2 + direction_vector_y ** 2)
                    direction_vector_x /= norm
                    direction_vector_y /= norm
                    robot_data += [
                        (x_big + w_big / 2, (y_big + h_big / 2), direction_vector_x, direction_vector_y)]

        n_possible_pos = len(robot_data)
        curr_t = time.time()
        curr_state = np.array(robot_data)
        idx = None
        if n_possible_pos == 1:
            self.prev_state = curr_state.squeeze()
            self.robot_states = np.vstack((self.robot_states, self.prev_state))
            self.t.append(curr_t)
            idx = 0
        elif n_possible_pos > 1 and self.prev_state is not None:
            d = np.linalg.norm(curr_state[:, :2] - self.prev_state[:2], axis=1)
            if self.pix2real(min(d)) < 0.1 or (curr_t-self.t[-1]) > 3:
                idx = np.argmin(d)
                curr_state = np.array(robot_data[idx])
                self.prev_state = curr_state.squeeze()
                self.robot_states = np.vstack((self.robot_states, self.prev_state))
                self.t.append(curr_t)

        if self.return_img:
            if idx is not None:
                x_big, y_big, w_big, h_big, x_small, y_small, w_small, h_small = robot_contour[idx]
                cv2.rectangle(Img, (x_big, y_big), (x_big + w_big, y_big + h_big), (0, 255,), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(Img, f"{self.robot_id} | size: {w_big * h_big}, {w_small * h_small}", (x_big + 14, y_big + 14), font, 0.7, (0, 0, 255),
                            2)  # +- 10 for better display
                lineThickness = 2
                cv2.line(Img, (int(robot_data[idx][0]), int(robot_data[idx][1])),
                         (int(robot_data[idx][0] + robot_data[idx][2] * 30),
                          int(robot_data[idx][1] + robot_data[idx][3] * 30)),
                         (0, 255, 0), lineThickness)
            return Img
        return

    def get_current_state(self):
        return self.prev_state
