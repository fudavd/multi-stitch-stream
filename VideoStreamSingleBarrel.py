import time

import cv2

from src.VideoStream import VideoStream
from src.VideoStream import ZMQHub
from src.Calibrate.Barrel import CalibrateBarrel


def run():
    with open("./secret/cam_paths.txt", "r") as file:
        paths = file.read().splitlines()
    ind = 1

    undistort = CalibrateBarrel(f'cam{ind}')
    resize = lambda im: cv2.resize(im, (960, 540))
    cam = VideoStream.VideoStreamSender(paths[ind], f'cam{ind}', [undistort.process_image, resize])
    hub = ZMQHub.ZMQHubReceiverThread(1, False)
    cam.start()
    hub.start()
    try:
        while not hub.stopped:
            # do_work(hub.latest_frames)
            time.sleep(1)
    except KeyboardInterrupt as e:
        print("Keyboard interrupt: CTR-C Detected. Closing threads")
    except Exception as e:
        print(e)

    while cam.thread.is_alive():
        cam.exit()
        time.sleep(0.1)
        print(f"{cam.thread.name}: {cam.cam_id}, closed!!!")

    while hub.thread.is_alive():
        hub.exit()
        time.sleep(0.1)
    print(f"{hub.thread.name}: ZMQ Hub, closed!!!")


if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
