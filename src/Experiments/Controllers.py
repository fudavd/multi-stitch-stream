import numpy as np
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng

pin_list = [17, 18, 27, 22, 23, 24, 10, 9, 25, 11, 8, 7, 5, 6, 12, 13, 16, 19, 20, 25, 21]

class CPG:
    def __init__(self, n_servos: int, weight_matrix = None, initial_state = None, limits = (-1.0, 1.0)):
        self.weight_matrix = weight_matrix
        self.initial_state = initial_state
        self.n_servos = n_servos
        self.lb = limits[0]
        self.ub = limits[1]
        self.rng = default_rng()
        self.rvs = stats.uniform(self.lb, self.ub - self.lb).rvs

    def random_network_mat(self, p_inter: float = 0.0):
        inter_connections = np.triu(random(self.n_servos, self.n_servos, density=p_inter,
                                          random_state=self.rng, data_rvs=self.rvs).A, 1)
        oscillator_connections = random(1, self.n_servos, density=1,
                                       random_state=self.rng, data_rvs=self.rvs).A
        weight_matrix = np.diag(oscillator_connections, self.n_servos)
        weight_matrix[1:self.n_servos, 1:self.n_servos] = inter_connections
        weight_matrix -= weight_matrix.T

    def load_mat(self, weight_matrix: np.array):
        self.weight_matrix = weight_matrix

    def create_config(self):
        pin_list = [17, 18, 27, 22, 23, 24, 10, 9, 25, 11, 8, 7, 5, 6, 12, 13, 16, 19, 20, 25, 21]
        initial_state = np.repeat([0.0, 1.0], self.n_servos).tolist()
        if self.initial_state is not None:
            initial_state =  self.initial_state

        weight_matrix = np.random.uniform(self.lb, self.ub, (self.n_servos*2, self.n_servos*2))
        if self.weight_matrix is not None:
            weight_matrix = self.weight_matrix

        config = {
            "controller_module": "revolve2.actor_controllers.cpg",
            "controller_type": "Cpg",
            "control_frequency": 64,
            "gpio": [{"dof": ind, "gpio_pin": pin_list[ind], "invert": False} for ind in range(self.n_servos)],
            "serialized_controller": {
                "state": initial_state,
                "num_output_neurons": self.n_servos,
                "weight_matrix": weight_matrix.tolist(),
            },
        }
        return config