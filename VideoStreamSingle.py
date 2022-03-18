import time
import cv2
from src.VideoStream import VideoStream
from src.VideoStream import ZMQHub
from src.Calibrate.Barrel_old import load_barrel_map
from src.Calibrate import create_transform_function
from src.Transform import MotionCapture


async def run():
    with open("./secret/cam_paths.txt", "r") as file:
        paths = file.read().splitlines()
    ind = 1
    map_x, map_y, roi = load_barrel_map(f'./Calibration_data/Barrel/cam{ind}.npz')
    t_func = create_transform_function(map_x, map_y)
    func = MotionCapture.MotionCapture(['green', 'blue', 'red'])
    cam = VideoStream.VideoStreamSender(paths[ind], f'cam{ind}', transform=[t_func, func.get_robot_pos])
    hub = ZMQHub.ZMQHubReceiverThread(1, True)
    cam.start()
    hub.start()
    try:
        cv2.namedWindow("LAB", cv2.WINDOW_KEEPRATIO)
        while not hub.stopped:
            dt, frame = await hub.wait_frame()
            cv2.imshow("LAB", frame)
            cv2.waitKey(1)
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
    import asyncio
    asyncio.run(run())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
