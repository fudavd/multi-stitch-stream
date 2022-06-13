import random

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import CpgRandom
import numpy as np


def create_default_robot(name: str, random_seed: int=420):
    body = Body()
    if name == "spider":
        body.core.left = ActiveHinge(np.pi / 2.0)
        body.core.left.attachment = Brick(-np.pi / 2.0)
        body.core.left.attachment.front = ActiveHinge(0.0)
        body.core.left.attachment.front.attachment = Brick(0.0)

        body.core.right = ActiveHinge(np.pi / 2.0)
        body.core.right.attachment = Brick(-np.pi / 2.0)
        body.core.right.attachment.front = ActiveHinge(0.0)
        body.core.right.attachment.front.attachment = Brick(0.0)

        body.core.front = ActiveHinge(np.pi / 2.0)
        body.core.front.attachment = Brick(-np.pi / 2.0)
        body.core.front.attachment.front = ActiveHinge(0.0)
        body.core.front.attachment.front.attachment = Brick(0.0)

        body.core.back = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment = Brick(0.0)
    if name == "gecko":
        body.core.left = ActiveHinge(0.0)
        body.core.left.attachment = Brick(0.0)

        body.core.right = ActiveHinge(0.0)
        body.core.right.attachment = Brick(0.0)

        body.core.back = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
        body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)
    if name == "babya":
        body.core.left = ActiveHinge(0.0)
        body.core.left.attachment = Brick(0.0)

        body.core.right = ActiveHinge(0.0)
        body.core.right.attachment = ActiveHinge(np.pi / 2.0)
        body.core.right.attachment.attachment = Brick(-np.pi / 2.0)
        body.core.right.attachment.attachment.front = ActiveHinge(0.0)
        body.core.right.attachment.attachment.front.attachment = Brick(0.0)

        body.core.back = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
        body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)
    body.finalize()
    rng = random.Random()
    rng.seed(random_seed)
    brain = CpgRandom(rng)
    robot = ModularRobot(body, brain)
    return robot


def show_grid_map(body, name: str="robot"):
    active_hinges_unsorted = body.find_active_hinges()
    active_hinge_map = {active_hinge.id: active_hinge for active_hinge in active_hinges_unsorted}
    _, dof_ids = body.to_actor()
    grid_map = np.array([body.grid_position(active_hinge_map[id]) for id in dof_ids]).astype(int)
    coord_min = grid_map.min(axis=0)
    coord_max = grid_map.max(axis=0)
    coord_range = coord_max-coord_min
    n_cols = coord_range[0]*2
    n_rows = coord_range[1]
    print_str = [[' '] * n_cols for _ in range(n_rows+1)]

    pin_list = ['17', '18', '27', '22', '23', '24', '10', '09', '25', '11', '08', '07', '05', '06', '12', '13', '16', '19', '20', '25', '21']
    print(f"Mapping {name} robot\n"
          f"HAT pin \t  | Coord")
    for ind, coord in enumerate(grid_map):
        col = coord[0] - coord_min[0]
        row = coord[1] - coord_min[1] + 1
        print(f'\t{pin_list[ind]}: ({coord[0]}\t, {coord[1]})')
        print_str[-row][2*col:2*(col+1)] = pin_list[ind]
    print_str[coord_min[1] - 1][-2*coord_min[0]:-2*(coord_min[0]-1)] = '▀▄'

    sub = '#'*int((n_cols - (len(name)))/2)
    print(f"\n{sub} {name} {sub}")
    for print_row in print_str:
        row_str = ''.join(print_row)
        print(row_str)

