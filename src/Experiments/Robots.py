import random
from typing import List, Tuple

from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
from revolve2.core.modular_robot.brains import make_cpg_network_structure_neighbour as mkcpg
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
    if name == "ant":
        body.core.left = ActiveHinge(0.0)
        body.core.left.attachment = Brick(0.0)

        body.core.right = ActiveHinge(0.0)
        body.core.right.attachment = Brick(0.0)

        body.core.back = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.left.attachment = Brick(0.0)
        body.core.back.attachment.right = ActiveHinge(0.0)
        body.core.back.attachment.right.attachment = Brick(0.0)

        body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
        body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
        body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
        body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)
    if name == "blokky":
        body.core.left = ActiveHinge(np.pi / 2.0)
        body.core.back = Brick(0.0)
        body.core.back.right = ActiveHinge(np.pi / 2.0)
        body.core.back.front = ActiveHinge(np.pi / 2.0)
        body.core.back.front.attachment = ActiveHinge(-np.pi / 2.0)
        body.core.back.front.attachment.attachment = Brick(0.0)
        body.core.back.front.attachment.attachment.front = Brick(0.0)
        body.core.back.front.attachment.attachment.front.right = Brick(0.0)
        body.core.back.front.attachment.attachment.front.right.left = Brick(0.0)
        body.core.back.front.attachment.attachment.front.right.front = Brick(0.0)
        body.core.back.front.attachment.attachment.right = Brick(0.0)
        body.core.back.front.attachment.attachment.right.front = Brick(0.0)
        body.core.back.front.attachment.attachment.right.front.right = Brick(0.0)
        body.core.back.front.attachment.attachment.right.front.front = ActiveHinge(0.0)
    if name == "park":
        body.core.back = ActiveHinge(np.pi / 2.0)
        body.core.back.attachment = ActiveHinge(-np.pi / 2.0)
        body.core.back.attachment.attachment = Brick(0.0)
        body.core.back.attachment.attachment.right = Brick(0.0)
        body.core.back.attachment.attachment.left = ActiveHinge(0.0)
        body.core.back.attachment.attachment.front = Brick(0.0)
        body.core.back.attachment.attachment.front.right = ActiveHinge(-np.pi / 2.0)
        body.core.back.attachment.attachment.front.front = ActiveHinge(-np.pi / 2.0)
        body.core.back.attachment.attachment.front.left = ActiveHinge(0.0)
        body.core.back.attachment.attachment.front.left.attachment = Brick(0.0)
        body.core.back.attachment.attachment.front.left.attachment.right = ActiveHinge(-np.pi / 2.0)
        body.core.back.attachment.attachment.front.left.attachment.front = Brick(0.0)
        body.core.back.attachment.attachment.front.left.attachment.front = ActiveHinge(0.0)
        body.core.back.attachment.attachment.front.left.attachment.front.attachment = Brick(0.0)
    body.finalize()
    _, dof_ids = body.to_actor()
    active_hinges_clean = body.find_active_hinges()

    active_hinge_map = {
        active_hinge.id: active_hinge for active_hinge in active_hinges_clean
    }
    active_hinges_sim = [active_hinge_map[id] for id in dof_ids]

    network_struct = mkcpg(active_hinges_sim)
    return body, network_struct


def show_grid_map(body, name: str="robot", hat_type="v1"):
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

    pin_lists = {"v1": ['17', '18', '27', '22', '23', '24', '10', ' 9', '25', '11', ' 8', ' 7', ' 5', ' 6', '12', '13', '16', '19', '20', '25', '21'],
                 "v2": [' 0', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '10', '11', '12', '13', '14', '15'],
                 "pca9685": [' 0', ' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '10', '11', '12', '13', '14', '15']}
    pin_list = pin_lists[hat_type]
    print(f"Mapping {name} robot\n"
          f"PIN #\t| Coord")
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

    active_hinges = [active_hinge_map[id] for id in dof_ids]
    connections = find_connections_full(active_hinges)
    dofs = np.array(dof_ids)
    pin_list_int = [int(pin) for pin in pin_list]
    l_mat = np.diag(pin_list_int[:len(dof_ids)])
    u_mat = np.zeros_like(l_mat)
    for connection in connections:
        row_n = np.where(dofs == connection[0].id)[0][0]
        col_n = np.where(dofs == connection[1].id)[0][0]
        u_mat[row_n, col_n] = pin_list[col_n]
    for ind in range(len(dof_ids)):
        print(f'\t{u_mat[ind,:]}  |{pin_list_int[ind]}|')


def find_connections_full(active_hinges: List[ActiveHinge]
) -> List[Tuple[ActiveHinge, ActiveHinge]]:
    # sort by id, will be used later when ignoring existing connections
    active_hinges.sort(key=lambda mod: mod.id)

    connections: List[Tuple[ActiveHinge, ActiveHinge]] = []
    for active_hinge in active_hinges:
        neighbours_all = active_hinge.neighbours(within_range=2)
        # ignore existing connections and neighbours that are not an active hinge
        neighbours = [
            neighbour
            for neighbour in neighbours_all
            if isinstance(neighbour, ActiveHinge)
        ]
        connections += zip([active_hinge] * len(neighbours), neighbours)
    return connections