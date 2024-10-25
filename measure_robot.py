import gc
import time
import numpy as np

from src.Experiments import MotionCapture
from revolve2.core.modular_robot import ModularRobot

from revolve2.core.modular_robot.brains import make_cpg_network_structure_neighbour as mkcpg, \
    BrainCpgNetworkNeighbourRandom
from src.Experiments.Controllers import CPG
from src.Experiments.Fitnesses import real_abs_dist, unwrapped_rot, signed_rot
from src.Experiments.Robots import show_grid_map
from src.VideoStream import ExperimentStream
import logging
from revolve2.core.rpi_controller_remote import connect

from src.utils.Measures import find_closest
from thirdparty.revolve2.standard_resources.revolve2.standard_resources import modular_robots


async def main() -> None:
    run_time = 60
    skills = ['gait', 'left', 'right']
    show_stream = False
    hat_version = "v1"
    id = 'spider'
    opt_type = 'we'
    body = modular_robots.get(id)
    show_grid_map(body, id, hat_version)

    if opt_type == 's0':
        run = 'n5_120.0'
        exp_folder = f'./experiment_data/real/{id}/{id}_{run}'
        weight_mat = np.load(exp_folder + '/weight_mat.npy', allow_pickle=True)
        weight_mats = np.array([weight_mat, weight_mat, weight_mat, ])
        f = np.load(f'{exp_folder}/fitness_full.npy', allow_pickle=True)
        x = np.load(f'{exp_folder}/x_full.npy', allow_pickle=True)

        init_idx = np.nanargmax(f, axis=0)
        print("expected fitnesses: \n",
              f[init_idx])
        initial_state = x[init_idx, :]
    elif opt_type == 'we':
        exp_folder = f'./experiment_data/real/{id}/transfer_weights/'
        _, dof_ids = body.to_actor()
        from random import Random
        rng = Random()
        rng.seed(5)
        brain = BrainCpgNetworkNeighbourRandom(rng)
        robot = ModularRobot(body, brain)
        _, controller = robot.make_actor_and_controller()
        active_hinges_clean = body.find_active_hinges()
        active_hinge_map = {active_hinge.id: active_hinge for active_hinge in active_hinges_clean}
        active_hinges_sim = [active_hinge_map[id] for id in dof_ids]
        network_struct = mkcpg(active_hinges_sim)
        genomes = []
        weight_mats = []
        for skill in skills:
            genome = np.loadtxt(f'{exp_folder}/weights_{skill}.txt', delimiter=', ')
            weight_mats.append(network_struct.make_connection_weights_matrix_from_params(genome))
            # genomes.append(np.repeat([0.5 * np.sqrt(2), 0.5 * np.sqrt(2)], len(dof_ids)))
            genomes.append(controller._state)
        initial_state = np.array(genomes)
        weight_mats = np.array(weight_mats)
    else:
        print("NON VALID")
        exit()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # async with connect("10.15.3.100", "pi", "raspberry") as conn:
    async with connect("10.15.2.155", "pi", "raspberry") as conn:
        print(f"Connection made with {id}")
        with open("./secret/cam_paths.txt", "r") as file:
            paths = file.read().splitlines()
        for ind in range( 2, initial_state.shape[0]):
            experiment = ExperimentStream.ExperimentStream(paths, show_stream=show_stream,
                                                           output_dir=exp_folder)
            print(f"Test {exp_folder} brain: skill {skills[ind]}")
            weight_mat = weight_mats[ind, :, :]
            n_servos = int(weight_mat.shape[0] / 2)
            brain = CPG(n_servos, weight_mat, initial_state[ind, :])
            config = brain.create_config(hat_version=hat_version)

            capture = MotionCapture.MotionCaptureRobot(f'{id}_{skills[ind]}', ["red", "green"],
                                                       return_img=show_stream)
            experiment.start_experiment([capture.store_img])
            robot_controller = asyncio.create_task(conn.run_controller(config, run_time))
            experiment_run = asyncio.create_task(experiment.stream())
            tasks = [experiment_run, robot_controller]
            # time.sleep(0.1)
            capture.clear_buffer()
            finished, unfinished = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in finished:
                print(task)
                start_time, log = robot_controller.result()

            experiment.close_stream()
            await experiment_run

            capture.post_process_img_buffer(exp_folder)
            np.save(f'{exp_folder}/{id}_{skills[ind]}/state_con', log)
            np.save(f'{exp_folder}/{id}_{skills[ind]}/start_con', start_time.timestamp())

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
            print(f'Retest for {skills[ind]}:\n'
                  f'Fitnesses: {fitnesses}\n')

            np.save(f'{exp_folder}/{id}_{skills[ind]}/fitnesses_trial', fitnesses)
            np.save(f'{exp_folder}/{id}_{skills[ind]}/x_trial', control_state[index])

            capture.clear_buffer()
            del capture
            del experiment
            gc.collect()
            await asyncio.sleep(1)

    print("Finished")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

