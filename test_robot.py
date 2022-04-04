import datetime
import threading
import time

import numpy as np

from src.Experiments import MotionCapture
# from src.VideoStream.ExperimentStream import ExperimentStream
from src.VideoStream import VideoStream, ZMQHub, close_stream, ExperimentStream
import logging
from revolve2.core.rpi_controller_remote import connect
import cv2

n_servos = 8
pin_list = [17, 18, 27, 22, 23, 24, 10, 9, 25, 11, 8, 7, 5, 6, 12, 13, 16, 19, 20, 25, 21]
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
config = {
    "controller_module": "revolve2.actor_controllers.cpg",
    "controller_type": "Cpg",
    "control_frequency": 60,
    "gpio": [{"dof": ind, "gpio_pin": pin_list[ind], "invert": False} for ind in range(n_servos)],
    "serialized_controller": {
        "state": np.repeat([0.0, 0.6], n_servos).tolist(),
        "num_output_neurons": n_servos,
        "weight_matrix": np.random.uniform(-1, 1, (n_servos*2, n_servos*2)).tolist(),
    },
}
# config = {
#     "controller_module": "revolve2.actor_controllers.cpg",
#     "controller_type": "Cpg",
#     "control_frequency": 10,
#     "gpio": [{"dof": 0, "gpio_pin": 17, "invert": False},
#              {"dof": 1, "gpio_pin": 18, "invert": False}],
#     "serialized_controller": {
#         "state": [0.0, 0.0]*2,
#         "num_output_neurons": 2,
#         "weight_matrix": np.zeros((2*2, 2*2)).tolist(),
#     },
# }


async def main() -> None:
    run_time = 10
    id = 'spider'

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    start_command_time = time.time()
    async with connect("10.15.3.100", "pi", "raspberry") as conn:
        start_time1, log1 = await conn.run_controller(config, 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
