import datetime
import os
import re
import threading
import time
from typing import List
import asyncio
import janus
import cv2
from threading import Thread
import imagezmq
import numpy as np
from src.Calibrate import create_single_remap


class ZMQHubReceiverThread:
    def __init__(self, n_stream,
                 verbose: bool = False,
                 merge_stream: bool = False,
                 snapshot_path: str = None,):
        self.hub = imagezmq.ImageHub()

        self.thread = Thread(target=self.update, args=())
        self.stop_event = threading.Event()
        self.thread.daemon = True
        self.stopped = False
        self.latest_frames = {}
        self.n_stream = n_stream
        self.verbose = verbose
        self.buffer = janus.Queue()

        self.h = -1
        self.w = -1
        self.start_t = time.time()
        self.output_dir = snapshot_path
        self.snapped = None
        if snapshot_path is not None:
            self.snapped = False
            self.output_dir = snapshot_path
            if not os.path.exists(self.output_dir):
                print(f"Creating output directory: {snapshot_path}")
                os.mkdir(self.output_dir)
        self.merge_stream = merge_stream
        try:
            if verbose:
                print("loading merge calibration data")
            cal_data = ["./Calibration_data/Stitch/cal_data.npz",
                        "./Calibration_data/Real/pan.npz"]
            x_maps = []
            y_maps = []
            for file in cal_data:
                if os.path.exists(file):
                    calibration = np.load(file)
                    x_maps.append(calibration["map_x"])
                    y_maps.append(calibration["map_y"])
            self.map_x, self.map_y = create_single_remap(x_maps, y_maps)
        except Exception as e:
            if verbose:
                print("Could not Load merge data")
                self.merge_stream = False

    def start(self):
        # start a thread to read frames from the file video stream
        if self.verbose:
            print(f"Start {self.thread.name}: ZMQ Hub")
        self.thread.start()
        return

    def initialize_connection(self):
        cam_stream_started = False
        cam_id = [None, "Never received image"]
        while not cam_stream_started and time.time() - self.start_t < 5:
            try:
                cam_id, frame = self.hub.recv_image()
                self.hub.send_reply(b'OK')
                cam_stream_started = True
                self.h, self.w = frame.shape[:2]
            except Exception as e:
                cam_id = [None, e.with_traceback()]
                time.sleep(0.1)
        if cam_id[1] != 'alive':
            print("Could not start stream", cam_id)
            self.stopped = True

    def update(self):
        self.initialize_connection()
        try:
            while not (self.stopped or self.n_stream == 0):
                self.start_t = time.time()
                cam_id, frame = self.hub.recv_image()

                if cam_id[1] != 'alive':
                    self.n_stream -= 1
                    if self.verbose:
                        print(cam_id)
                    if not self.merge_stream:
                        cv2.destroyWindow(cam_id[0])
                    continue
                cam_time = cam_id[2]
                if self.merge_stream:
                    frame = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
                    self.latest_frames["LAB"] = frame
                else:
                    self.latest_frames[cam_id[0]] = frame

                self.buffer.sync_q.put_nowait((cam_time, frame))
                # print(cam_id[0], self.buffer.sync_q.qsize(),
                #       round(time.time()-self.start_t, 3),
                #       round(cam_time - self.start_t, 3),
                #       round(merge_t - self.start_t, 3),
                #       round(time.time() - merge_t, 3))

                self.hub.send_reply(b'OK')
                if not self.snapped and time.time() - self.start_t > 10:
                    self.snapshot()
        except KeyboardInterrupt:
            time.sleep(1)
        except Exception as e:
            if self.verbose:
                print(e)

        cv2.destroyAllWindows()
        self.hub.send_reply(b'OK')
        self.buffer.close()
        self.hub.close()
        self.hub.zmq_socket.close()
        self.hub.zmq_context.term()
        self.stopped = True
        return

    async def wait_frame(self):
        return await self.buffer.async_q.get()

    def exit(self):
        # indicate that the thread should be stopped
        if self.verbose:
            print("Hub thread is still active:", self.thread.is_alive())
        self.stopped = True
        self.stop_event.set()
        self.thread.join()
        return self.thread.is_alive() and self.stop_event.is_set()

    def snapshot(self):
        frame_stack = np.zeros((self.h * self.n_stream, self.w, 3), dtype=np.uint8)
        for id, frame in self.latest_frames.items():
            outfile = os.path.join(self.output_dir, id + '.png')
            cv2.imwrite(outfile, frame)

        if self.merge_stream:
            img = cv2.remap(frame_stack, self.map_x, self.map_y, cv2.INTER_CUBIC)
            for transform in self.transform:
                img = transform(img)
            outfile = os.path.join(self.output_dir, 'panorama.png')
            cv2.imwrite(outfile, img)
        self.snapped = True
        print(f"snapshot(s) saved to {self.output_dir}")
