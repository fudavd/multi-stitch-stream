from . import *
from .ZMQHub import ZMQHubReceiverThread
from .VideoStream import VideoStreamSender
import time
from typing import List


def close_stream(cam_list: List[VideoStreamSender], hub: ZMQHubReceiverThread):
    for cam in cam_list:
        cam_alive = True
        while cam.thread.is_alive() or cam_alive:
            cam_alive = cam.exit()
            time.sleep(0.1)
        print(f"{cam.thread.name}: {cam.cam_id}, closed!!!")

    hub_alive = True
    while hub.thread.is_alive() or hub_alive:
        hub_alive = hub.exit()
        time.sleep(0.1)
    print(f"{hub.thread.name}: ZMQ Hub, closed!!!")
