import datetime
import os.path
import threading
import time
from typing import List

import cv2
import numpy as np

from src.Calibrate import create_transform_function
from src.Calibrate.Stitcher import PairStitcher, MultiStitcher
from src.VideoStream import VideoStream, close_stream
from src.VideoStream import ZMQHub
from src.Calibrate.Barrel import CalibrateBarrel, load_barrel_map
from src.utils.Filters import draw_grid
from src.utils.utils import PanoramaImage, Image, search_file_list


def undistort_barrel(path: str, name: str):
    """
    Calibrate videostream for barrel distortion

    :param path: path to access camera stream
    :param name: camera name which would be used to store calibration data
    :return:
    """
    resize = lambda im: cv2.resize(im, (960, 540))
    hub = ZMQHub.ZMQHubReceiverThread(1, False)
    undistort = CalibrateBarrel(name)
    cam = VideoStream.VideoStreamSender(path, name, [undistort.process_image, resize])
    cam.start()
    hub.start()
    try:
        while not hub.stopped:
            time.sleep(1)
            if os.path.exists(f"./Calibration_data/Barrel/{name}.npz"):
                cam.exit()
                hub.exit()
                break
    except KeyboardInterrupt as e:
        print("Keyboard interrupt: CTR-C Detected. Closing threads")
    except Exception as e:
        print(e)
        exit()
    close_stream([cam], hub)


def multistream(paths: List[str], merge_stream=False, snapshot=None):
    """
    Stream multiple camera's at once

    :param paths: list of paths to access camera streams
    :param merge_stream: if True streams will be merged into 1 single window
    :param snapshot:
    :return:
    """
    cam_list = []
    if merge_stream:
        hub = ZMQHub.ZMQHubReceiverThread(len(paths), verbose=True, merge_stream=True, snapshot_path=snapshot)
    else:
        hub = ZMQHub.ZMQHubReceiverThread(len(paths), verbose=True, merge_stream=False, snapshot_path=snapshot)
    hub.start()
    try:
        for ind, path in enumerate(paths):
            map_x, map_y, roi = load_barrel_map(f'./Calibration_data/Barrel/cam{ind}.npz')
            t_func = create_transform_function(map_x, map_y)
            cam = VideoStream.VideoStreamSender(path, f'cam{ind}', transform=[t_func])
            cam.start()
            cam_list.append(cam)
        cv2.namedWindow("LAB", cv2.WINDOW_KEEPRATIO)
        while not hub.stopped and hub.snapped is not True:
            if not hub.buffer.empty():
                dt, frame = hub.buffer.get_nowait()
                print(hub.buffer.qsize(), dt)
                cv2.imshow("LAB", frame)
            cv2.waitKey(1)
    except KeyboardInterrupt as e:
        print("Keyboard interrupt: CTR-C Detected. Closing threads")
    except Exception as e:
        print(e)
    close_stream(cam_list, hub)


def stitch_snapshots(paths: List[str], output_dir: str, verbose=False):
    """
    Stitch snapshots of multiple stream into a single panorama.
     - First, a single panorama estimate is created which provides a global Homography list
     - Second, sequential stitching errors/artifacts are corrected by iterative re-stitching of a pseudorama
     - Finally, x/y mapping and final global homographies are saved in the output_dir
    :param paths: List of paths to the snapshots
    :param output_dir: output directory
    :param verbose: show intermediate pseudorama and final panorama
    :return:
    """
    images = [Image(f'img{ind}', path) for ind, path in enumerate(paths)]

    multi_stitch = MultiStitcher(images, verbose=False)
    multi_stitch.stitch()
    multi_stitch.panorama.stitch_panorama("pareto")
    global_h = multi_stitch.panorama.get_global_h()

    images_p = multi_stitch.panorama.images
    repetitions = 100
    n_images = len(paths)
    indices = np.arange(n_images)
    for counter in range(repetitions):
        ind = counter % n_images
        shuffle = (indices[1:n_images] + ind) % n_images
        # shuffle = np.random.choice(shuffle, len(shuffle), replace=False)
        images_shuffled = [images_p[ii] for ii in shuffle]
        global_h_shuffled = [global_h[ii] for ii in shuffle]
        pseudorama = PanoramaImage("pseudo", images_shuffled)
        pseudorama.set_global_h(global_h_shuffled)
        pseudorama.h, pseudorama.w = multi_stitch.panorama.image.shape[:2]
        pseudorama.stitch_panorama("linear_blend")
        result = pseudorama.panorama
        if verbose == True:
            cv2.imshow('test', cv2.resize(result, (720, 405)))
            cv2.waitKey(1)
        gap_img = images_p[ind]
        stitcher = PairStitcher([pseudorama, gap_img], save_res=True)
        stitcher.stitch(cov_min=0, cov_rat=0.1)
        global_h[ind] = np.mean((stitcher.H, global_h[ind]), axis=0)

    global_h_final = {}
    global_h_test = []
    for image in images:
        for ind, image_p in enumerate(images_p):
            if np.all(image_p.image == image.image):
                global_h_final[image.name] = global_h[ind]
                global_h_test.append(global_h[ind])
                break

    res = PanoramaImage("result", images)
    res.set_global_h(global_h_test)
    res.output_dir = output_dir
    res.h, res.w = multi_stitch.panorama.image.shape[:2]
    res.stitch_panorama("pareto")
    result = res.panorama
    cv2.imshow('test', cv2.resize(result, (720, 405)))
    cv2.waitKey(1)

    my_details = {
        'map_x': res.map_x,
        'map_y': res.map_y,
        'H_dict': global_h_final
    }
    np.savez(os.path.join(output_dir, "cal_data"), **my_details)


def calibrate_panorama(output_dir="./Calibration_data/Real/"):
    global coord
    def register_mouse_click(event, x, y, flags, params):
        global coord
        if event == cv2.EVENT_LBUTTONDOWN:
            coord = np.array([x, y])
    real_coord = np.loadtxt(output_dir + "real_coord.csv", delimiter=",")
    src_im = cv2.imread("Calibration_data/Stitch/panorama.png")
    n_coord = len(real_coord)
    img_points = np.zeros((n_coord, 2), np.int32)

    resolution = np.array(src_im.shape[:2][::-1])
    offset = 0.05*resolution
    ratio = np.mean(resolution*0.9/np.max(real_coord[:, :2], axis=0))
    coord = np.array([-1, -1])
    prev_coord = np.array([-1, -1])
    n = 0
    cv2.namedWindow("LAB", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("LAB", src_im)
    cv2.waitKey()
    if not os.path.exists(output_dir+"pan_coord.csv"):
        while n < n_coord:
            pred_coord = offset+ratio*real_coord[n, :2]
            cv2.putText(src_im, f"Point {n}, {real_coord[n, :2]}", pred_coord.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("LAB", src_im)
            cv2.setMouseCallback("LAB", register_mouse_click)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
            if np.any(coord != prev_coord):
                if np.linalg.norm (coord - prev_coord) <= 50:
                    cv2.drawMarker(src_im, (img_points[n][0], img_points[n][1]), (0, 0, 255), 0, 10)
                    cv2.drawMarker(src_im, (img_points[n][0], img_points[n][1]), (0, 255, 0), 0, 10)
                    img_points[n-1, :] = coord
                    prev_coord = coord
                    continue
                img_points[n, :] = coord
                prev_coord = coord
                cv2.drawMarker(src_im, (img_points[n][0], img_points[n][1]), (0, 255, 0), 0, 10)
                n +=1
        np.savetxt(output_dir+"pan_coord.csv", img_points, delimiter=",", fmt='%1.5e')
    else:
        img_points = np.loadtxt(output_dir+"pan_coord.csv", delimiter=",")
    img_points = img_points.squeeze()
    rms, mtx, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(np.float32([real_coord]),
                                                             np.float32([img_points]), src_im.shape[::-1][1:],
                                                             None, None)
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coefs, src_im.shape[:2], 1, src_im.shape[:2])
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist_coefs, None, new_mtx, src_im.shape[::-1][1:], 5)
    my_details = {
        'map_x': map_x,
        'map_y': map_y,
        'roi': roi,
        'H': (mtx, new_mtx)
    }
    np.savez(os.path.join(output_dir, "pan"), **my_details)
    src_im2 = draw_grid(src_im, line_color=(255, 0, 0), line_dst=75)
    img_points2, _ = cv2.projectPoints(real_coord, rvecs[0], tvecs[0], mtx, dist_coefs)
    cv2.imshow("LAB", src_im2)
    cv2.waitKey()
    pan_out = cv2.remap(src_im2, map_x, map_y, cv2.INTER_CUBIC)
    for ind in range(len(img_points2)):
        cv2.drawMarker(pan_out, (int(img_points[ind][0]), int(img_points[ind][1])), (255, 255, 255), 0, 10)
        cv2.drawMarker(pan_out, (int(img_points2[ind][0][0]), int(img_points2[ind][0][1])), (125, 0, 255), 0, 50)
    pan_out = draw_grid(pan_out, line_dst=75)
    error = cv2.norm(img_points, img_points2.squeeze(), cv2.NORM_L1) / len(img_points2)
    print(f"Reprojection error: {error}")
    cv2.imshow("LAB", pan_out)
    cv2.waitKey()
    cv2.imwrite(output_dir + "/pan_out.jpg", pan_out)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with open("./secret/cam_paths.txt", "r") as file:
        paths = file.read().splitlines()

    remove_barrel = False
    if remove_barrel:
        for ind, path in enumerate(paths):
            undistort_barrel(path, f'cam{ind}')

    stitch_cameras = False
    stitch_output_dir = os.path.join("./Calibration_data", "Stitch", f"snapshot_{datetime.date.today().isoformat()}")
    if not os.path.exists(stitch_output_dir):
        os.mkdir(stitch_output_dir)
    if stitch_cameras:
        if not os.path.exists(stitch_output_dir + f'cam{len(paths)}.png'):
            multistream(paths, snapshot=True)
        snapshots = search_file_list(stitch_output_dir, '.png')
        stitch_snapshots(snapshots, "./Calibration_data/Stitch")

    undistort_pan = False
    real_output_dir = './Calibration_data/Real/'
    if undistort_pan:
        if not os.path.exists("./Calibration_data/Stitch/panorama.png"):
            multistream(paths, merge_stream=True, snapshot="./Calibration_data/Stitch/panorama.png")
        calibrate_panorama(real_output_dir)

    check_full_pan = True
    if check_full_pan:
        multistream(paths, merge_stream=True)

