import copy

import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
from utils.CPG_network import RK45

pin_list = {"v1": [17, 18, 27, 22, 23, 24, 10, 9, 25, 11, 8, 7, 5, 6, 12, 13, 16, 19, 20, 25, 21],
            "v2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "pca9685": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}

def gram_schmidt(vectors):
    """
    Performs Gram-Schmidt orthogonalization on a set of vectors.
    """
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)

class CPG:
    def __init__(self, n_servos: int, weight_matrix=None, initial_state=None, limits=(-1.0, 1.0)):
        if weight_matrix is not None:
            self.weight_matrix = weight_matrix
        else:
            self.set_rand_weights()
        if initial_state is None:
            self.initial_state = np.repeat([0.0, 1.0], n_servos)
        else:
            self.initial_state = initial_state
        self.n_servos = n_servos
        self.lb = limits[0]
        self.ub = limits[1]
        self.rng = default_rng()
        self.rvs = stats.uniform(self.lb, self.ub - self.lb).rvs

    def set_rand_weights(self, p_inter: float = 0.0):
        inter_connections = np.triu(random(self.n_servos, self.n_servos, density=p_inter,
                                           random_state=self.rng, data_rvs=self.rvs).A, 1)
        oscillator_connections = random(1, self.n_servos, density=1,
                                        random_state=self.rng, data_rvs=self.rvs).A
        weight_matrix = np.diag(oscillator_connections, self.n_servos)
        weight_matrix[1:self.n_servos, 1:self.n_servos] = inter_connections
        weight_matrix -= weight_matrix.T
        self.weight_matrix = weight_matrix

    def create_config(self, hat_version="v1"):
        if hat_version == "pca9685":
            con_type = "pca9685"
        else:
            con_type = "hat"+hat_version
        config = {
            "controller_module": "revolve2.actor_controllers.cpg",
            "hardware":  con_type,
            "controller_type": "CpgActorController",
            "control_frequency": 64,
            "gpio": [{"dof": ind, "gpio_pin": pin_list[hat_version][ind], "invert": False} for ind in range(self.n_servos)],
            "serialized_controller": {
                "state": self.initial_state.tolist(),
                "num_output_neurons": self.n_servos,
                "weight_matrix": self.weight_matrix.tolist(),
            "dof_ranges": np.ones(self.n_servos).tolist(),
            },
        }
        return config

    def set_weights(self, weight_matrix: np.ndarray):
        self.weight_matrix = weight_matrix

    def set_initial_state(self, initial_state: np.ndarray):
        self.initial_state = initial_state


class CPG_feedback:
    def __init__(self, n_servos: int, weight_matrix, initial_states, window_time = 60, cooldown_time = 10, dt=0.1):
        self.dt = dt
        self.n_skills = len(initial_states)
        self.n_states = n_servos*2
        self.n_servos = n_servos
        self.state_shape = (self.n_states, 1)
        self.window_size = int(window_time/dt)

        self.A = weight_matrix
        self.y = np.array(initial_states[0]).reshape(self.n_states, 1)
        self.heading_error = 0
        self.cur_skill = 0
        self.cur_ind = 0
        self.prev_skill = 0
        self.prev_ind = 0
        self.switch = False

        self.cooldown = int(cooldown_time/dt)
        self.skills = np.zeros((self.n_skills, self.n_states, self.window_size))
        for ind, initial_state in enumerate(initial_states):
            y = np.array(initial_state)[:, np.newaxis]
            for _ in range(self.window_size-1):
                y_t = RK45(y[:, -1], self.A, self.dt).reshape(self.state_shape)
                y = np.hstack((y, y_t))
            self.skills[ind] = np.array(y)

        self.transitions = np.repeat(np.tile(np.arange(self.window_size)+1,
                                             (self.n_skills, self.n_skills, 1))[:,:,:,np.newaxis], 2, axis=-1)
        for skill_ind in range(self.n_skills):
            self.transitions[skill_ind, :, :, 0] = np.repeat([[0],[1],[2]], self.window_size, axis=1)

        for skill_i in range(self.n_skills):
            current_skill = self.skills[skill_i, :, :]

            for skill_j in range(self.n_skills):
                transition_skill = self.skills[skill_j, :, :]
                min_error = np.inf
                for t in range(self.window_size, self.cooldown, -1):
                    next_ind = t
                    next_skill = skill_i
                    x_diff = (transition_skill[:n_servos, :t].T -
                              current_skill[:n_servos, min(t, self.window_size-1)])
                    x_pert = np.linalg.norm(x_diff, axis=1)/n_servos
                    # x_pert = np.sum(np.abs(x_diff), axis=1)/n_servos

                    x_dot_i = (RK45(current_skill[:, min(t, self.window_size-1)], self.A, self.dt) -
                               current_skill[:, min(t, self.window_size-1)])
                    x_dot_j = RK45(transition_skill, self.A, self.dt) - transition_skill
                    x_dot_diff = x_dot_j[:n_servos].T - x_dot_i[:n_servos]
                    x_dot_pert = np.linalg.norm(x_dot_diff, axis=1)/n_servos
                    # x_dot_pert = np.sum(np.abs(x_dot_diff), axis=1)

                    perturbation = x_pert[:t - self.cooldown] + x_dot_pert[:t - self.cooldown] / self.dt
                    min_ind = np.argmin(perturbation)
                    if perturbation[min_ind] < min_error:
                        next_ind = min_ind
                        next_skill = skill_j
                        min_error = perturbation[min_ind]
                        # print((next_skill, next_ind), " NEW MIN ERROR:", min_error)
                    else:
                        if skill_i == skill_j:
                            if skill_i == 1:
                                min_error = -1.0005
                            else:
                                min_error *= 1.005
                        else:
                            min_error = perturbation[min_ind]

                    transition_ind = (next_skill, next_ind)
                    self.transitions[skill_i, skill_j, t-1, :] = transition_ind

    def get_dof_targets(self):
        next_skill = 0
        if self.heading_error > 0.5:
            next_skill = 1
        if self.heading_error < -0.5:
            next_skill = 2

        self.prev_skill, self.prev_ind = (self.cur_skill, self.cur_ind)
        self.cur_skill, self.cur_ind = self.transitions[self.cur_skill, next_skill, self.cur_ind, :]

        next_target_state = self.skills[self.cur_skill, :, self.cur_ind].reshape(self.state_shape)

        if self.cur_ind != self.prev_ind + 1 or self.prev_skill != self.cur_skill:
            self.switch = True
            prev_next_state = RK45(self.y[:, -1].reshape(self.state_shape), self.A, self.dt)
            norm_next_state = np.linalg.norm(prev_next_state)

            perturbed_state = copy.deepcopy(prev_next_state)
            perturbed_state[self.n_servos:] = copy.deepcopy(next_target_state[self.n_servos:])
            next_target_state = RK45(perturbed_state, self.A, self.dt)
            next_target_state = next_target_state/np.linalg.norm(next_target_state)*norm_next_state
        else:
            self.switch = False
            next_target_state = (copy.deepcopy(RK45(self.y[:, -1].reshape(self.state_shape), self.A, self.dt))/self.dt + copy.deepcopy(next_target_state))/(1 + 1/self.dt)
            # next_target_state = RK45(self.y[:, -1].reshape(self.state_shape), self.A, self.dt)

        self.y = np.hstack((self.y, next_target_state))
        return next_target_state[:self.n_servos].squeeze().tolist()

    def step(self, time):
        return None