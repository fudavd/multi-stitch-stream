import re
import threading
from threading import Thread, Event
import time
from typing import List

import imagezmq
import numpy as np
from imutils.video import VideoStream


class VideoStreamSender:
    def __init__(self, paths: List[str], stream_id: str, transform: List = []):
        self.streams = []
        for ind, path in enumerate(paths):
            cap = VideoStream(path)
            self.streams.append((ind, cap.start()))
        self.stopped = False
        self.transform = transform

        self.sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
        self.cam_id = stream_id

        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.stop_event = threading.Event()
        self.h = None
        self.w = None
        self.n_streams = len(self.streams)

    def start(self):
        # start a thread to read frames from the file video stream
        print(f"Start {self.thread.name}: {self.cam_id}")
        self.thread.start()
        return

    def update(self):
        # keep looping infinitely
        stop_string = 'stream loading error'
        frame = self.streams[0][1].read()
        self.h, self.w = frame.shape[:2]
        frame_stack = np.zeros((self.h * self.n_streams, self.w, 3), dtype=np.uint8)

        try:
            while not self.stopped:
                curr_time = time.time()
                for ind, cam in self.streams:
                    frame = cam.read()
                    frame = self.transform[ind](frame)
                    frame_stack[self.h * ind:self.h * (ind + 1), :] = frame
                self.sender.send_image((self.cam_id, 'alive', curr_time), frame_stack)
            stop_string = "recv. stop signal"
        except Exception as e:
            stop_string = e.with_traceback()

        self.sender.send_image((self.cam_id, stop_string), np.zeros(4))
        time.sleep(1)
        self.sender.close()
        self.sender.zmq_socket.close()
        self.sender.zmq_context.term()
        for ind, cam in self.streams:
            cam.stop()
        return

    def exit(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stop_event.set()
        self.thread.join()
        return self.thread.is_alive() and self.stop_event.is_set()
