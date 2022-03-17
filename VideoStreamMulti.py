import time

import cv2

from src.Transform import MotionCapture
from src.Calibrate import create_transform_function
from src.Calibrate.Barrel_old import load_barrel_map
from src.VideoStream import VideoStream, ZMQHub, close_stream


async def run():
    with open("./secret/cam_paths.txt", "r") as file:
        paths = file.read().splitlines()

    cam_list = []
    capture = MotionCapture.MotionCaptureRobot("spider", ["red", "green"], return_img=True).log_robot_pos
    hub = ZMQHub.ZMQHubReceiverThread(len(paths), verbose=True, merge_stream=True)
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
            # time.sleep(0.01)
            dt, frame = await hub.wait_frame()
            frame = capture(frame)
            cv2.imshow("LAB", frame)
            cv2.waitKey(1)
            print(hub.buffer.qsize())

    except KeyboardInterrupt as e:
        print("Keyboard interrupt: CTR-C Detected. Closing threads")
    except Exception as e:
        print(e)
    close_stream(cam_list, hub)


if __name__ == '__main__':
    import asyncio
    asyncio.run(run())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
