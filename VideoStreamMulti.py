import time

import cv2

from src.Experiments import MotionCapture
from src.Calibrate import create_transform_function
from src.Calibrate.Barrel_old import load_barrel_map
from src.VideoStream import VideoStream, ZMQHub, close_stream


async def run():
    with open("./secret/cam_paths.txt", "r") as file:
        paths = file.read().splitlines()

    cam_list = []
    show_stream = True
    capture = MotionCapture.MotionCaptureRobot("spider", ["red", "green"], return_img=show_stream).log_robot_pos
    hub = ZMQHub.ZMQHubReceiverThread(1, verbose=True, merge_stream=True)
    hub.start()
    try:
        t_funcs = []
        for ind, path in enumerate(paths):
            map_x, map_y, roi = load_barrel_map(f'./Calibration_data/Barrel/cam{ind}.npz')
            t_funcs.append(create_transform_function(map_x, map_y))

        cam = VideoStream.VideoStreamSender(paths, f'frame_stack', transform=t_funcs)
        cam.start()
        cam_list.append(cam)
        if show_stream:
            cv2.namedWindow("LAB", cv2.WINDOW_KEEPRATIO)
        while not hub.stopped and hub.snapped is not True:
            dt, frame = await hub.wait_frame()
            frame = capture(frame)
            if show_stream:
                cv2.imshow("LAB", frame)
                cv2.waitKey(1)
            # print(hub.buffer.async_q.qsize(), dt)

    except KeyboardInterrupt as e:
        print("Keyboard interrupt: CTR-C Detected. Closing threads")
    except Exception as e:
        print(e)
    close_stream(cam_list, hub)


if __name__ == '__main__':
    import asyncio
    asyncio.run(run())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
