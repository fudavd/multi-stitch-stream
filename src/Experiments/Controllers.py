import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot

pin_list = {"v1": [17, 18, 27, 22, 23, 24, 10, 9, 25, 11, 8, 7, 5, 6, 12, 13, 16, 19, 20, 25, 21],
            "v2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "pca9685": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}


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

