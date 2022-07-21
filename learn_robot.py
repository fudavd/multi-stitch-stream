import gc
import os.path
import time
import numpy as np

from src.Experiments import MotionCapture
from src.Experiments.Robots import create_default_robot, show_grid_map
from src.Experiments.Controllers import CPG
from src.Experiments.Learners import DifferentialEvolution
from src.Experiments.Fitnesses import real_abs_dist, unwrapped_rot, signed_rot
from src.VideoStream import ExperimentStream
import logging
from revolve2.core.rpi_controller_remote import connect
from revolve2.core.modular_robot.brains import make_cpg_network_structure_neighbour as mkcpg

from src.utils.Measures import find_closest

def rand_CPG_network(num_dofs, inter_con_density: float = 0.5):
    from scipy.sparse import random
    from numpy.random import default_rng
    from scipy import stats
    rng = default_rng()
    rvs = stats.uniform(-1, 2).rvs
    inter_connection = np.triu(random(num_dofs, num_dofs, density=inter_con_density,
                                      random_state=rng, data_rvs=rvs).A, 1)
    oscillator_connection = random(1, num_dofs, density=1,
                                      random_state=rng, data_rvs=rvs).A
    oscillator_connection = np.insert(oscillator_connection, range(1, num_dofs), 0)
    weight_matrix = np.diag(oscillator_connection, 1)
    weight_matrix[::2, ::2] = inter_connection
    weight_matrix -= weight_matrix.T
    return weight_matrix

async def main() -> None:
    run_time = 120
    window_time = 60
    n_runs = 5
    run = 0
    show_stream = False
    error = False
    id = 'ant'
    exp_folder = f'./experiment_data/{id}'

    body, network_struct = create_default_robot(id, random_seed=np.random.randint(1000))
    show_grid_map(body, id)

    matrix_weights = np.random.uniform(-1, 1, network_struct.num_params)
    weight_mat = network_struct.make_weight_matrix_from_params(matrix_weights)

    n_servos = network_struct.num_states
    if os.path.isfile(exp_folder + '/weight_mat.npy'):
        weight_mat = np.load(exp_folder + '/weight_mat.npy', allow_pickle=True)
    np.save(exp_folder + '/weight_mat', weight_mat)
    initial_state = np.random.uniform(-1, 1, (n_servos, n_runs))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )
    c = []
    x = []
    f = []
    for pre_ind in range(run):
        c_trial = np.load(f'{exp_folder}/{id}_{pre_ind}/state.npy', allow_pickle=True)
        f_trial = np.load(f'{exp_folder}/{id}_{pre_ind}/fitnesses_trial.npy', allow_pickle=True)
        x_trial = np.load(f'{exp_folder}/{id}_{pre_ind}/x_trial.npy', allow_pickle=True)
        c.append(c_trial)
        f.append(np.array(f_trial).squeeze())
        x.append(x_trial)


    async with connect("10.15.3.100", "pi", "raspberry") as conn:
        print(f"Connection made with {id}")
        with open("./secret/cam_paths.txt", "r") as file:
            paths = file.read().splitlines()
        while run < n_runs and not error:
            experiment = ExperimentStream.ExperimentStream(paths, show_stream=show_stream,
                                                           output_dir=f'./experiment_data/{id}')
            print(f"Set new brain for run {run +1}/{n_runs}")
            brain = CPG(n_servos, weight_mat, initial_state[:, run])
            config = brain.create_config()

            capture = MotionCapture.MotionCaptureRobot(f'{id}_{run}', ["red", "green"], return_img=show_stream)
            experiment.start_experiment([capture.capture_aruco])
            robot_controller = asyncio.create_task(conn.run_controller(config, run_time))
            experiment_run = asyncio.create_task(experiment.stream())
            tasks = [experiment_run, robot_controller]

            finished, unfinished = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            time.sleep(0.1)
            for task in finished:
                print(task)
                start_time, log = robot_controller.result()

            experiment.close_stream()
            await experiment_run

            # save results: capture_data, controller_data, start_data
            time.sleep(0.1)
            # capture.post_process_img_buffer(exp_folder)
            capture.save_results(exp_folder)
            np.save(f'{exp_folder}/{id}_{run}/state_con', log)
            np.save(f'{exp_folder}/{id}_{run}/start_con', start_time.timestamp())

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

            index = capture_t < (run_time - window_time)

            x.append(control_state[index])

            n_samples = index.sum()
            f_trial = []
            for ind in range(n_samples):
                t_rel = capture_t[ind]
                window_idx = (t_rel <= capture_t) & (capture_t <= t_rel + window_time)
                f_dist = real_abs_dist(capture_state[window_idx][:, :2]).squeeze()
                f_angle = signed_rot(capture_state[window_idx][:, 2:])
                fitnesses = np.array([f_dist, f_angle, -f_angle])
                if np.isnan(f_angle):
                    fitnesses[1:] = -np.inf
                f_trial.append(fitnesses)
            print(np.nanmax(f_trial, axis=0))
            f.append(np.array(f_trial).squeeze())
            c.append(capture_state)
            np.save(f'{exp_folder}/{id}_{run}/fitnesses_trial', f_trial)
            np.save(f'{exp_folder}/{id}_{run}/x_trial', control_state[index])
            run += 1

            capture.clear_buffer()
            del capture
            del experiment
            gc.collect()
            await asyncio.sleep(1)

    np.save(f'{exp_folder}/capture_full', np.array(np.vstack(c)))
    np.save(f'{exp_folder}/fitness_full', np.array(np.vstack(f)))
    np.save(f'{exp_folder}/x_full', np.array(np.vstack(x)))
    print("FINISHED")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
