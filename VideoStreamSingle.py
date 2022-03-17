import time
from src.VideoStream import VideoStream
from src.VideoStream import ZMQHub
from src.Calibrate.Barrel_old import load_barrel_map
from src.Calibrate import create_transform_function
from src.Transform import MotionCapture


def run():
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
        while not hub.stopped:
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
