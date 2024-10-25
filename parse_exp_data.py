import os
import numpy as np
import time
import dateutil
import src.utils.utils as utils
import matplotlib.pyplot as plt

from src.Experiments.Fitnesses import unwrapped_rot, real_abs_dist, signed_rot
from src.utils.Measures import find_closest


trial = 1
robot = 'spider'
controller_state = np.load(f'./experiment_data/{robot}/{robot}_{trial}/state_con.npy', allow_pickle=True)
controller_time = np.load(f'./experiment_data/{robot}/{robot}_{trial}/start_con.npy', allow_pickle=True)
position_state = np.load(f'./experiment_data/{robot}/{robot}_{trial}/state.npy', allow_pickle=True)
position_time = np.load(f'./experiment_data/{robot}/{robot}_{trial}/t.npy', allow_pickle=True)

print("hello", len(controller_state))
state_rob = position_state
t_rob = position_time
figure, ax = plt.subplots(2, 1)
ax[0].plot(state_rob[:,0], state_rob[:,1])
ax[0].set_aspect('equal', 'box')
t_con = np.array([con['timestamp'] for con in controller_state])
state_con = np.array([con['serialized_controller']['state'] for con in controller_state])
ax[1].plot(t_rob - controller_time.flatten(), state_rob[:, :2])
ax[1].plot((t_con - t_con[0])/1000, state_con*500+850)
ax[1].legend(['x_pos', 'y_pos'])
figure.show()
figure.savefig("test.pdf")

state_rob = position_state
t_rob = position_time
figure, ax = plt.subplots(2, 1)
ax[0].plot(state_rob[:,0], state_rob[:,1])
ax[0].set_aspect('equal', 'box')


run_time = 120
capture_t = (t_rob - controller_time.flatten()).squeeze()
capture_state = state_rob[(0 < capture_t) & (capture_t <= run_time)]
capture_t = capture_t[(0 < capture_t) & (capture_t <= run_time)]
control_t = (t_con - t_con[0])/1000
index = find_closest(control_t, capture_t).squeeze()
control_state = state_con[index]
control_t = control_t[index]
window_time = 60

index = capture_t < (run_time - window_time)
n_samples = index.sum()
f_trial = []
print(n_samples)
for ind in range(n_samples):
    t_rel = capture_t[ind]
    window_idx = (t_rel <= capture_t) & (capture_t <= t_rel + window_time)
    f_dist = real_abs_dist(capture_state[window_idx][:, :2]).squeeze()
    f_angle = unwrapped_rot(np.arctan2(capture_state[window_idx][:, 3],
                                       capture_state[window_idx][:, 2])[:, np.newaxis])
    f_angle2 = signed_rot(capture_state[window_idx][:, 2:])
    fitnesses = np.array([f_dist, f_angle, -f_angle, f_angle2, -f_angle2])
    if np.isnan(f_angle):
        fitnesses[1:] = -np.inf
    f_trial.append(fitnesses)
max_ind = np.nanargmax(f_trial, axis=0)
np.save('initial', control_state[max_ind, :])
print(max_ind)

figure, ax = plt.subplots(2, 1)
ax[0].plot(capture_state[:,0], capture_state[:,1])
ax[0].set_aspect('equal', 'box')
ax[0].scatter(capture_state[max_ind, 0], capture_state[max_ind, 1])
ax[1].plot(capture_t, capture_state)
ax[1].plot(control_t, control_state*500+850)
ax[1].legend(['x_pos', 'y_pos'])
# ax[1].scatter(capture_t[max_ind], capture_state[max_ind, :])
# ax[1].scatter(capture_t[max_ind], capture_state[max_ind, :])
figure.show()
figure.savefig("test2.pdf")

figure, ax = plt.subplots(1, 1)
ax.plot(control_t, control_state)
controller_state_ = np.load(f'./experiment_data/spider/spider_gait/state_con.npy', allow_pickle=True)
t_con_ = np.array([con['timestamp'] for con in controller_state_])
control_t_ = (t_con_ - t_con_[0]) / 1000 + control_t[max_ind[0]]
control_state_ = np.array([con['serialized_controller']['state'] for con in controller_state_])
ax.plot(control_t_, control_state_ +5)
figure.show()

print(control_state[max_ind[0]:max_ind[0]+20,:] - control_state_[:20,:])
# print(len(control_state), len(control_state_left), )
