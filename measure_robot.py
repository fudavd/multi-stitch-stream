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
genome = np.random.uniform(-1, 1, (n_servos + 4))
inter_weights = genome[:n_servos]
weight_matrix = np.diag(inter_weights, n_servos)
weight_matrix -= weight_matrix.T

initial_state = np.random.uniform(-1, 1, (n_servos*2))
config = {
    "controller_module": "revolve2.actor_controllers.cpg",
    "controller_type": "Cpg",
    "control_frequency": 60,
    "gpio": [{"dof": ind, "gpio_pin": pin_list[ind], "invert": False} for ind in range(n_servos)],
    "serialized_controller": {
        "state": np.repeat([0.0, 0.6], n_servos).tolist(),
        "num_output_neurons": n_servos,
        "weight_matrix": weight_matrix.tolist(),
    },
}


async def main() -> None:
    run_time = 60
    n_runs = 1
    run = 0
    show_stream = False
    error = False
    id = 'spider'

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    async with connect("10.15.3.100", "pi", "raspberry") as conn:
        print(f"Connection made with {id}")
        with open("./secret/cam_paths.txt", "r") as file:
            paths = file.read().splitlines()
        experiment = ExperimentStream.ExperimentStream(paths, show_stream=show_stream, output_dir=f'./experiment_data/{id}')
        while run < n_runs and not error:
            capture = MotionCapture.MotionCaptureRobot(f'{id}_{run}', ["red", "green"], return_img=show_stream)
            experiment.start_experiment([capture.log_robot_pos])
            robot_controller = asyncio.create_task(conn.run_controller(config, run_time))
            experiment_run = asyncio.create_task(experiment.stream())
            tasks = [experiment_run, robot_controller]
            start_command_time = time.time()
            finished, unfinished = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                start_time, log = robot_controller.result()

            experiment.stop_experiment()
            await experiment_run
            capture.save_results(f'./experiment_data/{id}')
            np.save(f'./experiment_data/{id}/{id}_{run}/state_con', log)
            np.save(f'./experiment_data/{id}/{id}_{run}/start_con', np.array([start_time, start_command_time]))

            state_con = np.empty((0, len(log[0]['serialized_controller']['state'])))
            t_con = np.empty((0, 1))
            for sample in log:
                t_con = np.vstack((t_con, sample['timestamp']))
                state_con = np.vstack((state_con, sample['serialized_controller']['state']))

            run += 1
    experiment.stop_stream()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
