import threading
import time
from src.Transform import MotionCapture
from src.Calibrate import create_transform_function
from src.Calibrate.Barrel_old import load_barrel_map
from src.VideoStream import VideoStream, ZMQHub, close_stream
import logging
from revolve2.core.rpi_controller_remote import connect


def run_camera():
    paths = []
    with open("./secret/cam_paths.txt", "r") as file:
        for line in file:
            paths.append(line)

    cam_list = []
    map_x, map_y, roi = load_barrel_map('./Calibration_data/Real/pan.npz')
    pan_cal = create_transform_function(map_x, map_y, roi)
    capture = MotionCapture.MotionCaptureRobot("spider", ["red", "green"]).log_robot_pos
    hub = ZMQHub.ZMQHubReceiverThread(len(paths), verbose=True, merge_stream=True)
    hub.start()
    try:
        for ind, path in enumerate(paths):
            map_x, map_y, roi = load_barrel_map(f'./Calibration_data/Barrel/cam{ind}.npz')
            t_func = create_transform_function(map_x, map_y)
            cam = VideoStream.VideoStreamSender(path, f'cam{ind}', transform=[t_func, ])
            cam.start()
            cam_list.append(cam)
        while not hub.stopped and hub.snapped is not True:
            time.sleep(1)
    except KeyboardInterrupt as e:
        print("Keyboard interrupt: CTR-C Detected. Closing threads")
    except Exception as e:
        print(e)
    close_stream(cam_list, hub)


config = {
    "controller_module": "revolve2.actor_controllers.cpg",
    "controller_type": "Cpg",
    "control_frequency": 10,
    "gpio": [{"dof": 0, "gpio_pin": 17, "invert": False}],
    "serialized_controller": {
        "state": [0.707, 0.707],
        "num_output_neurons": 1,
        "weight_matrix": [[0.0, 0.5], [-0.5, 0.0]],
    },
}


async def main(camera_thread) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    start_command_time = time.time()
    async with connect("15.15.3.100", "pi", "raspberry") as conn:
        start_time1, log1 = await conn.run_controller(config, 60)

    camera_thread.start()

if __name__ == "__main__":
    running = threading.Event()
    running.set()
    camera_thread = threading.Thread(target=run_camera, args=())

    import asyncio

    asyncio.run(main(camera_thread))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
