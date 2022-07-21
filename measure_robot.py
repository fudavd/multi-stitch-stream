import gc
import time
import numpy as np

from src.Experiments import MotionCapture
from src.Experiments.Robots import create_default_robot, show_grid_map
from src.Experiments.Controllers import CPG
from src.Experiments.Fitnesses import real_abs_dist, unwrapped_rot, signed_rot
from src.VideoStream import ExperimentStream
import logging
from revolve2.core.rpi_controller_remote import connect

from src.utils.Measures import find_closest


async def main() -> None:
    run_time = 60
    skills = ['gait', 'left', 'right']
    show_stream = False
    id = 'ant'
    exp_folder = f'./experiment_data/{id}'
    weight_mat = np.load(exp_folder + '/weight_mat.npy', allow_pickle=True)
    f = np.load(f'{exp_folder}/fitness_full.npy', allow_pickle=True)
    x = np.load(f'{exp_folder}/x_full.npy', allow_pickle=True)

    init_idx = np.nanargmax(f, axis=0)
    print("expected fitnesses: \n",
          f[init_idx])
    initial_state = x[init_idx, :]
    n_servos = int(weight_mat.shape[0]/2)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    async with connect("10.15.3.100", "pi", "raspberry") as conn:
        print(f"Connection made with {id}")
        with open("./secret/cam_paths.txt", "r") as file:
            paths = file.read().splitlines()
        for skill in range(initial_state.shape[0]):
            experiment = ExperimentStream.ExperimentStream(paths, show_stream=show_stream,
                                                           output_dir=f'./experiment_data/{id}')
            print(f"Test {exp_folder} brain: skill {skills[skill]}")
            brain = CPG(n_servos, weight_mat, initial_state[skill, :])
            config = brain.create_config()

            capture = MotionCapture.MotionCaptureRobot(f'{id}_{skills[skill]}', ["red", "green"], return_img=show_stream)
            experiment.start_experiment([capture.store_img])
            robot_controller = asyncio.create_task(conn.run_controller(config, run_time))
            experiment_run = asyncio.create_task(experiment.stream())
            tasks = [experiment_run, robot_controller]
            time.sleep(0.1)
            finished, unfinished = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in finished:
                print(task)
                start_time, log = robot_controller.result()

            experiment.close_stream()
            await experiment_run

            capture.post_process_img_buffer(exp_folder)
            np.save(f'{exp_folder}/{id}_{skills[skill]}/state_con', log)
            np.save(f'{exp_folder}/{id}_{skills[skill]}/start_con', start_time.timestamp())

            state_con = np.empty((0, len(log[0]['serialized_controller']['state'])))
            t_con = np.empty((0, 1))
            for sample in log:
                t_con = np.vstack((t_con, sample['timestamp']))
                state_con = np.vstack((state_con, sample['serialized_controller']['state']))

            capture_t = np.array(capture.t) - start_time.timestamp()
            capture_state = capture.robot_states
            capture_state = capture_state[(0 < capture_t) & (capture_t <= run_time), :]
            capture_t = capture_t[(0 < capture_t) & (capture_t <= run_time)]

            control_t = (t_con.flatten() - t_con[0, 0]) / 1000
            index = find_closest(control_t, capture_t)
            control_state = state_con[index]

            index = capture_t < run_time

            f_dist = real_abs_dist(capture_state[:, :2]).squeeze()
            f_angle2 = signed_rot(capture_state[:, 2:])
            fitnesses = np.array([f_dist, f_angle2, -f_angle2])
            print(f'Retest for {skills[skill]}: {f[init_idx[skill]]}\n'
                  f'Fitnesses: {fitnesses}\n')

            np.save(f'{exp_folder}/{id}_{skills[skill]}/fitnesses_trial', fitnesses)
            np.save(f'{exp_folder}/{id}_{skills[skill]}/x_trial', control_state[index])

            capture.clear_buffer()
            del capture
            del experiment
            gc.collect()
            await asyncio.sleep(1)

    print("Finished")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
