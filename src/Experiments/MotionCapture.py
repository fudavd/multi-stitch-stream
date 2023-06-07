import time
from typing import List

import cv2
import numpy as np

from ..utils.Geometry import pix2meter
from ..utils.Filters import mask_colors
from ..utils.Measures import obtain_color_range


class MotionCapture:
    def __init__(self, colors=None, ):
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
    def __init__(self, robot_id: str, colors: List[str], return_img=False, verbose=True):
        self.verbose = verbose
        self.return_img = return_img
        self.robot_id = robot_id
        self.colors = colors
        self.contour_colors = (obtain_color_range([colors[0]]), obtain_color_range([colors[1]]))
        self.tolerance = 5
        self.pix2real = pix2meter
        self.robot_states = np.empty((0, 4))  # (x, y, x_v, y_v)
        self.t = []
        self.prev_state = None
        self.img_buffer = []

    def capture_rgb(self, img, time_stamp=None):
        Img = img.copy()
        Img_r = cv2.resize(Img.copy(), (int(Img.shape[1] / 2), int(Img.shape[0] / 2)))
        contours_area = []
        # Img_r = cv2.medianBlur(Img_r, 3)
        hsv = cv2.cvtColor(Img_r, cv2.COLOR_BGR2HSV)
        for c_range in self.contour_colors:
            mask = mask_colors(hsv, c_range)
            # mask = cv2.GaussianBlur(mask,  (5, 5), 0)
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
                    robot_data += [
                        (x_big + w_big / 2, (y_big + h_big / 2), direction_vector_x, direction_vector_y)]

        n_possible_pos = len(robot_data)
        if time_stamp is None:
            curr_t = time.time()
        else:
            curr_t = time_stamp
        curr_state = np.array(robot_data)
        idx = None
        if n_possible_pos == 1:
            self.prev_state = curr_state.squeeze()
            self.robot_states = np.vstack((self.robot_states, self.prev_state))
            self.t.append(curr_t)
            idx = 0
        elif n_possible_pos > 1 and self.prev_state is not None:
            d = np.linalg.norm(curr_state[:, :2] - self.prev_state[:2], axis=1)
            if self.pix2real(min(d)) < 0.5 or (curr_t - self.t[-1]) > 3:
                idx = np.argmin(d)
                curr_state = np.array(robot_data[idx])
                self.prev_state = curr_state.squeeze()
                self.robot_states = np.vstack((self.robot_states, self.prev_state))
                self.t.append(curr_t)

        if self.return_img:
            # if idx is not None:
            #     x_big, y_big, w_big, h_big, x_small, y_small, w_small, h_small = robot_contour[idx]
            #     cv2.rectangle(Img, (x_big, y_big), (x_big + w_big, y_big + h_big), (0, 255,), 3)
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(Img, f"{self.robot_id} | size: {w_big * h_big}, {w_small * h_small}",
            #                 (x_big + 14, y_big + 14), font, 0.7, (0, 0, 255), 2)  # +- 10 for better display
            #     lineThickness = 2
            #     cv2.line(Img, (int(robot_data[idx][0]), int(robot_data[idx][1])),
            #              (int(robot_data[idx][0] + robot_data[idx][2] * 30),
            #               int(robot_data[idx][1] + robot_data[idx][3] * 30)),
            #              (0, 255, 0), lineThickness)
            return Img
        return

    def capture_aruco(self, img, time_stamp=None):
        # detects aruco aruco_tags and returns list of ids and coords of centre
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray.copy(), (int(gray.shape[1] / 2), int(gray.shape[0] / 2)))
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_parameters)

        robot_data = []
        if ids is not None:
            for i in range(len(ids)):
                total_x, total_y = 0, 0
                # find average of corner positions to give the centre position
                corner_pos = (np.array(corners[i][0]) * 2)
                x, y = np.mean(corner_pos, axis=0).astype(int)
                dir_vec = np.mean(corner_pos[:2, :] - corner_pos[2:, :], axis=0).astype(int)
                robot_data += [(x, y, dir_vec[0], dir_vec[1])]

        n_possible_pos = len(robot_data)
        if time_stamp is None:
            curr_t = time.time()
        else:
            curr_t = time_stamp
        curr_state = np.array(robot_data)
        idx = None
        if n_possible_pos == 1:
            self.prev_state = curr_state.squeeze()
            self.robot_states = np.vstack((self.robot_states, self.prev_state))
            self.t.append(curr_t)
            idx = 0
        elif n_possible_pos > 1 and self.prev_state is not None:
            d = np.linalg.norm(curr_state[:, :2] - self.prev_state[:2], axis=1)
            if self.pix2real(min(d)) < 0.5 or (curr_t - self.t[-1]) > 3:
                idx = np.argmin(d)
                curr_state = np.array(robot_data[idx])
                self.prev_state = curr_state.squeeze()
                self.robot_states = np.vstack((self.robot_states, self.prev_state))
                self.t.append(curr_t)
        if self.return_img:
            if idx is not None:
                x, y, x_vec, y_vec = robot_data[idx]
                cv2.rectangle(img, (int(corner_pos[:, 0].min()), int(corner_pos[:, 1].min())),
                              (int(corner_pos[:, 0].max()), int(corner_pos[:, 1].max())), (0, 255,), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, f"{ids} | ({x}, {y})",
                            (x + 14, y + 14), font, 0.7, (0, 0, 255), 2)  # +- 10 for better display
                lineThickness = 2
                cv2.line(img, (int(x), int(y)),
                         (int(x + x_vec),
                          int(y + y_vec)),
                         (0, 255, 0), lineThickness)
            # try:
            #     if curr_t - self.t[-1] >= 0.5:
            #         cv2.imwrite(os.path.join('experiment_data', 'camera', f'{curr_t}.jpg'), img)
            # except:
            #     return
            return img
        return

    def store_img(self, img, time_stamp=None):
        if time_stamp is None:
            curr_t = time.time()
        else:
            curr_t = time_stamp
        self.img_buffer.append((curr_t, img))

    def get_current_state(self):
        return self.prev_state

    def save_results(self, dir=''):
        if len(self.robot_states) == 0 or len(self.t) == 0 or len(self.robot_states) != len(self.t):
            if self.verbose:
                print(f'Could not save PositionCapture len: robot states {len(self.robot_states)}, t {len(self.t)}')
        elif os.path.exists(dir):
            if not os.path.exists(os.path.join(dir, self.robot_id)):
                os.mkdir(os.path.join(dir, self.robot_id))
            np.save(os.path.join(dir, self.robot_id, "state", ), np.array(self.robot_states))
            np.save(os.path.join(dir, self.robot_id, "t", ), np.array(self.t).reshape((len(self.t), 1)))

    def save_img_buffer(self, dir=''):
        path = os.path.join(dir, self.robot_id, 'images')
        try:
            os.makedirs(path)
        except OSError as error:
            print("Directory '%s' can not be created")
        for t, img in self.img_buffer:
            cv2.imwrite(os.path.join(path, f'{t}.jpg'), img)

    def post_process_img_buffer(self, dir=''):
        path = os.path.join(dir, self.robot_id, 'images')
        try:
            os.makedirs(path)
            print(path, 'is made')
        except OSError as error:
            print(f"Directory {path} can not be created")
        print(f'Parse img buffer: len {len(self.img_buffer)}')
        for t, img in self.img_buffer:
            cv2.imwrite(os.path.join(path, f'{t}.jpg'), img)
            self.capture_aruco(img, t)

    def clear_buffer(self):
        self.img_buffer.clear()

from cv2 import aruco

# aruco dictionary and parameters
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_parameters = aruco.DetectorParameters_create()
aruco_parameters.adaptiveThreshWinSizeMin = 3
aruco_parameters.adaptiveThreshWinSizeMax = 18
aruco_parameters.adaptiveThreshWinSizeStep = 3
aruco_parameters.minMarkerPerimeterRate = 0.01
aruco_parameters.maxMarkerPerimeterRate = 4
aruco_parameters.polygonalApproxAccuracyRate = 0.1
aruco_parameters.perspectiveRemovePixelPerCell = 10
# define an empty custom dictionary for markers of size 4
aruco_dict = aruco.custom_dictionary(0, 3, 1)
aruco_dict.bytesList = np.empty(shape = (1, 2, 4), dtype = np.uint8)
mybits = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 1]], dtype = np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)

# f = plt.figure()
# plt.imshow(aruco.drawMarker(aruco_dict, 0, 128), cmap='gray')
# plt.axis(False)
# f.savefig("custom_aruco_" + str(i) + ".pdf", bbox_inches='tight')

def MotionCaptureRobot_Aruco(image):
    """
    returns tag_list: a list of lists. Each sub-list is 3 elements, the tag number, x and y positions
    [ [tag_number,x,y] , [tag_number,x,y] , ... ]
    """
    # detects aruco aruco_tags and returns list of ids and coords of centre
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray.copy(), (int(gray.shape[1] / 2), int(gray.shape[0] / 2)))
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_parameters)

    # goes through list of tag ids and finds the centre point
    tag_list = []

    if ids is not None:
        robot_data = []
        for i in range(len(ids)):
            total_x, total_y = 0, 0
            # find average of corner positions to give the centre position
            corner_pos = (np.array(corners[i][0]) * 2)
            x, y = np.mean(corner_pos, axis=0).astype(int)
            dir_vec = np.mean(corner_pos[:2, :] - corner_pos[2:, :], axis=0).astype(int)
            robot_data += [(x, y, dir_vec[0], dir_vec[1])]

        x, y, x_vec, y_vec = robot_data[-1]
        cv2.rectangle(image, (int(corner_pos[:, 0].min()), int(corner_pos[:, 1].min())),
                      (int(corner_pos[:, 0].max()), int(corner_pos[:, 1].max())), (0, 255,), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"{ids} | ({x}, {y})",
                    (x + 14, y + 14), font, 0.7, (0, 0, 255), 2)  # +- 10 for better display
        lineThickness = 2
        cv2.line(image, (int(x), int(y)),
                 (int(x + x_vec),
                  int(y + y_vec)),
                 (0, 255, 0), lineThickness)

    # returns list
    return image
