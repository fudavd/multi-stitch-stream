from threading import Thread
import time
import imagezmq
import numpy as np
from imutils.video import VideoStream


class VideoStreamSender:
    def __init__(self, path: str, stream_id: str, transform: list = []):
        cap = VideoStream(path)
        self.stream = cap.start()
        self.stopped = False
        self.transform = transform

        self.sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
        self.cam_id = stream_id

        # intialize thread
        self.thread = Thread(target=self.update, args=())

    def start(self):
        # start a thread to read frames from the file video stream
        print(f"Start {self.thread.name}: {self.cam_id}")
        self.thread.start()
        return

    def update(self):
        # keep looping infinitely
        stop_string = 'stream loading error'
        try:
            while not self.stopped:
                frame = self.stream.read()
                for transform in self.transform:
                    frame = transform(frame)
                self.sender.send_image((self.cam_id, 'alive'), frame)
                self.last_image = frame
        except Exception as e:
            stop_string = e.with_traceback()

        self.sender.send_image((self.cam_id, stop_string, frame), np.zeros(4))
        self.sender.close()
        self.sender.zmq_socket.close()
        self.sender.zmq_context.term()
        self.stream.stop()
        return

    def exit(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
        return self.thread.is_alive()
