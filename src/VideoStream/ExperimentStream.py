import asyncio
import threading
import time
import cv2
from typing import List
from src.VideoStream import VideoStream, ZMQHub, close_stream
from src.Calibrate import create_transform_function
from src.Calibrate.Barrel_old import load_barrel_map


class ExperimentStream:
    def __init__(self, paths: List[str], verbose=False, show_stream=False, output_dir="./experiment_data/"):
        self.running = threading.Event()
        self.running.set()
        self.hub = ZMQHub.ZMQHubReceiverThread(1, verbose=True, merge_stream=True)
        self.hub.start()
        cam_list = []
        t_funcs = []
        for ind, path in enumerate(paths):
            map_x, map_y, roi = load_barrel_map(f'./Calibration_data/Barrel/cam{ind}.npz')
            t_funcs.append(create_transform_function(map_x, map_y))

        cam = VideoStream.VideoStreamSender(paths, f'frame_stack', transform=t_funcs)
        cam.start()
        cam_list.append(cam)
        self.cam_list = cam_list
        self.exp_func = []

        self.exp_folder = output_dir
        self.verbose = verbose
        self.start_time = None
        self.stopped = False
        self.show_stream = show_stream

    def await_stream(self):
        self.stream()
        return True

    def close_stream(self):
        close_stream(self.cam_list, self.hub, self.running)
        self.stopped = True
        return True

    async def clear_buffer(self):
        while not self.hub.buffer.sync_q.empty():
            await self.hub.wait_frame()
            cv2.waitKey(1)
        if self.verbose:
            print("buffer cleared")

    async def stream(self):
        if self.show_stream:
            cv2.namedWindow("LAB", cv2.WINDOW_KEEPRATIO)
        try:
            while not self.stopped:
                time_stamp, frame = await self.hub.wait_frame()
                for func in self.exp_func:
                    frame = func(frame, time_stamp)
                if self.show_stream:
                    cv2.imshow("LAB", frame)
                cv2.waitKey(1)
        except KeyboardInterrupt as e:
            print("Keyboard interrupt: CTR-C Detected. Closing threads")
        except Exception as e:
            print(e)
        if self.verbose:
            print("stream stopped")

    def start_experiment(self, func=[]):
        self.exp_func = func
        self.stopped = False
        self.start_time = time.time()

    def pause_experiment(self):
        if self.verbose:
            print("Pausing stream")
        self.stopped = True
        self.exp_func = []

    def stop_stream(self):
        close_stream(self.cam_list, self.hub, self.running)