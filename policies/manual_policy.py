import copy
import operator
import pickle as pkl
import pprint
from time import sleep

import click
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from storehouse.environment.env_storehouse import Storehouse
from tqdm import tqdm

STEPS = 100000
SLEEP_TIME = 0.00
VISUAL = True
AUGMENT_FACTOR = 6


def store_item(env: Storehouse, waiting_items: list):
    target_entrypoint = [ep.position for ep in waiting_items][0]
    state, reward, done, info = env._step(target_entrypoint)
    if VISUAL:
        print(f"General action: store_item\nAction: {target_entrypoint}\nReward: {reward}\n{info}")
        env.render()
    if done:
        if VISUAL:
            print("######################")
        env.reset()
    sleep(SLEEP_TIME)

    target_cell = None
    for ii in range(1, env.grid.shape[0] - 1):
        for jj in range(1, env.grid.shape[1] - 1):
            if env.grid[ii][jj] == 0:
                target_cell = (ii, jj)
                break
        else:
            continue
        break
    if target_cell is None:
        done = True
    else:
        state, reward, done, info = env._step(target_cell)
        if VISUAL:
            print(f"General action: store_item\nAction: {target_cell}\nReward: {reward}\n{info}")
            env.render()
    if done:
        if VISUAL:
            print("######################")
        env.reset()
    sleep(SLEEP_TIME)


def deliver_from_entrypoints(env: Storehouse, available_types: list, entrypoints_with_items: list):
    boxes = [
        ep.material_queue[0]["material"]
        for ep in entrypoints_with_items
        if ep.material_queue[0]["material"].type in available_types
    ]
    da_box = boxes[0]
    state, reward, done, info = env._step(da_box.position)
    if VISUAL:
        print(f"General action: deliver from entrypoint \nAction: {da_box.position}\nReward: {reward}\n{info}")
        env.render()
    if done:
        if VISUAL:
            print("######################")
        env.reset()
    sleep(SLEEP_TIME)

    state, reward, done, info = env._step(env.outpoints.outpoints[0])
    if VISUAL:
        print(f"General action: deliver_item\nAction: {env.outpoints.outpoints[0]}\nReward: {reward}\n{info}")
        env.render()
    if done:
        if VISUAL:
            print("######################")
        env.reset()
    sleep(SLEEP_TIME)


def deliver_item(env: Storehouse, available_types: list):
    oldest_box = max(
        (
            material
            for material in env.material.values()
            if material.type in available_types and material.position not in env.restricted_cells
        ),
        key=operator.attrgetter("age"),
    )

    state, reward, done, info = env._step(oldest_box.position)
    if VISUAL:
        print(f"General action: deliver_item\nAction: {oldest_box.position}\nReward: {reward}\n{info}")
        env.render()
    if done:
        if VISUAL:
            print("######################")
        env.reset()
    sleep(SLEEP_TIME)

    state, reward, done, info = env._step(env.outpoints.outpoints[0])
    if VISUAL:
        print(f"General action: deliver_item\nAction: {env.outpoints.outpoints[0]}\nReward: {reward}\n{info}")
        env.render()
    if done:
        if VISUAL:
            print("######################")
        env.reset()
    sleep(SLEEP_TIME)


def wait(env: Storehouse):
    action = (0, 1)
    state, reward, done, info = env._step(action)
    render(env, "wait", action, reward, info)
    s = check_reset(env, done)
    state = state if s is None else s
    return (state,)


def initial_human_policy(env: Storehouse, state: np.array):
    try:
        ready_to_consume_types = [order["type"] for order in env.outpoints.delivery_schedule if order["timer"] == 0]
    except IndexError:
        ready_to_consume_types = []
    try:
        entrypoints_with_items = [
            ep for ep in [ep for ep in env.entrypoints if len(ep.material_queue) > 0] if ep.material_queue[0]["timer"] == 0
        ]
    except IndexError:
        entrypoints_with_items = []
    if any(entrypoints_with_items):
        store_item(env, entrypoints_with_items)
    elif (
        any(ready_to_consume_types)
        and len(
            [
                True
                for box in env.material.values()
                if box.type in ready_to_consume_types and box.position not in env.restricted_cells
            ]
        )
        > 0
    ):
        deliver_item(env, ready_to_consume_types)
    else:
        wait(env)


def enhanced_human_policy(env: Storehouse, state: np.array):
    try:
        ready_to_consume_types = [order["type"] for order in env.outpoints.delivery_schedule if order["timer"] == 0]
    except IndexError:
        ready_to_consume_types = []
    try:
        entrypoints_with_items = [
            ep for ep in [ep for ep in env.entrypoints if len(ep.material_queue) > 0] if ep.material_queue[0]["timer"] == 0
        ]
    except IndexError:
        entrypoints_with_items = []
    if any(ready_to_consume_types):
        if len(
            [
                True
                for box in env.material.values()
                if box.type in ready_to_consume_types and box.position not in env.restricted_cells
            ]
        ):  # Items in the grid
            deliver_item(env, ready_to_consume_types)
        elif len(
            [
                ep.material_queue[0]["material"].type
                for ep in entrypoints_with_items
                if ep.material_queue[0]["material"].type in ready_to_consume_types
            ]
        ):  # Items in the EPs
            deliver_from_entrypoints(env, ready_to_consume_types, entrypoints_with_items)
        elif entrypoints_with_items:
            store_item(env, entrypoints_with_items)
        else:
            wait(env)
    elif any(entrypoints_with_items):
        store_item(env, entrypoints_with_items)
    else:
        wait(env)


def int_to_type(material: int) -> str:
    """
    Converts 0, 1, 2,... to A, B, C...
    """
    return chr(ord("A") + material)


def decode_material(encoded_material: list) -> set:
    return {int_to_type(material) for material in encoded_material}


def denormalize_out_state(state: float, num_types: int) -> int:
    return int(state / 255 * (2 ** num_types - 1))


def deserialize_out_state(state: int) -> list:
    powers = []
    ii = 1
    while ii <= state:
        if ii & state:
            powers.append(ii)
        ii <<= 1
    return list(np.log2(powers).astype(int))


def get_ready_to_consume_items_from_state(s: np.array, out_position: list, num_types: int) -> list:
    box_grid = s[0]
    outpoint_state = box_grid[out_position]
    if outpoint_state == 0:
        return []
    denormalized_out_state = denormalize_out_state(outpoint_state, num_types)
    wanted_material_encoded = deserialize_out_state(denormalized_out_state)
    return decode_material(wanted_material_encoded)


def denormalize_type(encoded_type: int, num_types: int) -> str:
    return int_to_type(int(encoded_type / 255 * num_types) - 1)


class EntrypointBox:
    def __init__(self, pos, age, type):
        self.pos = pos
        self.age = age
        self.type = type


def get_ep_with_items_from_state(s: np.array, ep_position: list, num_types: int) -> dict:
    box_grid = s[0]
    age_grid = s[1]
    ep_state = {pos: box_grid[pos] for pos in ep_position}
    if not any(ep_state):
        return []
    return [EntrypointBox(pos, age_grid[pos], denormalize_type(ep, num_types)) for pos, ep in ep_state.items() if ep]


# def filter_restricted_boxes(grid: np.array, num_types: int) -> dict:
#     adjacent_cells_delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#     boxes = {}
#     for row in range(1, grid.shape[0] - 1):
#         for col in range(1, grid.shape[1] - 1):
#             flag = True
#             for delta in adjacent_cells_delta:
#                 adjacent_cell = tuple(map(operator.add, (row, col), delta))
#                 if grid[adjacent_cell] == 0:
#                     flag = False
#                     break
#             if not flag and grid[row][col] > 0:  # Not restricted_cell and cell with item
#                 boxes[(row, col)] = denormalize_type(grid[row][col], num_types)
#     return boxes  # form {pos: type}


def get_ready_boxes_in_grid(s: np.array, ready_types: set, num_types: int) -> list:
    box_grid = s[0]
    available_boxes = {}
    for ii, row in enumerate(box_grid):
        for jj, val in enumerate(row):
            if val > 0:
                available_boxes[(ii, jj)] = denormalize_type(val, num_types)
    return [pos for pos, box in available_boxes.items() if box in ready_types]


def filter_age_grid(age_grid: np.array, ready_boxes: list, entrypoints, outpoints) -> np.array:
    grid = age_grid
    mask = np.zeros(grid.shape)
    for box in ready_boxes:
        if check_reachable(grid, (1, 0), box, entrypoints, outpoints):
            mask[box] = 1
    return np.multiply(grid, mask)


def get_oldest_box(age_grid: np.array, ready_boxes: list, entrypoints, outpoints) -> tuple:
    grid = filter_age_grid(age_grid, ready_boxes, entrypoints, outpoints)
    oldest_boxes = get_max_position(grid)
    return oldest_boxes[0]


def render(env: Storehouse, general_action: str, action: tuple, reward: float, info: dict):
    if VISUAL:
        pp = pprint.PrettyPrinter(depth=4, sort_dicts=False)
        pp.pprint({"Info": info, "General action": general_action, "Action": action, "Reward": reward})
        env.render()


def check_reset(env, done) -> np.array:
    if done:
        if VISUAL:
            print("######################")
        return env.reset()
    return None


def take_item_from_grid(state, ready_boxes, entrypoints, outpoints) -> np.array:
    age_grid = state[1]
    return get_oldest_box(age_grid, ready_boxes, entrypoints, outpoints)


def deliver_box(outpoint_position):
    return outpoint_position


def take_item_from_ep(ep: list) -> int:  # Return action
    return max(ep, key=operator.attrgetter("age")).pos


def deposit_item_in_grid(state: np.array, entrypoints, outpoints) -> np.array:
    age_grid = state[1]
    agent_grid = state[2]
    target_cell = find_target_cell(age_grid, entrypoints, outpoints)
    return (4, 0) if target_cell is None else target_cell


def idle():
    return (0, 1)


def prepare_grid(matrix: np.array, start: tuple, end: tuple, whitelist: list = []) -> Grid:
    prepared_matrix = np.array(matrix)
    prepared_matrix[start] = 0
    prepared_matrix[end] = 0
    for cell in whitelist:
        prepared_matrix[cell] = 0
    return Grid(matrix=np.negative(prepared_matrix) + 1)


def check_reachable(matrix: np.array, start_cell: tuple, end_cell: tuple, entrypoints: list, outpoints: list) -> bool:
    grid = prepare_grid(
        copy.deepcopy(matrix),
        start_cell,
        end_cell,
        whitelist=entrypoints + outpoints,
    )
    start = grid.node(*reversed(start_cell))
    end = grid.node(*reversed(end_cell))
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)
    return len(path)


def find_target_cell(
    grid: np.array,
    entrypoints,
    outpoints,
) -> tuple:  # TODO: Change to  pathfinding
    target_cell = None
    for ii in range(1, grid.shape[0] - 1):
        for jj in range(1, grid.shape[1] - 1):
            if grid[ii][jj] == 0:
                if not check_reachable(grid, (1, 0), (ii, jj), entrypoints, outpoints):
                    continue
                target_cell = (ii, jj)
                break
        else:
            continue
        break
    return target_cell


def get_max_position(grid: np.array) -> list:
    max_value_unsorted = np.where(grid == np.amax(grid))
    return list(zip(max_value_unsorted[0], max_value_unsorted[1]))


def calculate_item_type(box_grid: np.array, agent_grid: np.array, num_types: int) -> list:
    agent_position = get_max_position(agent_grid)[0]
    assert type(agent_position) is tuple
    return denormalize_type(box_grid[agent_position], num_types)


def get_agent_item_type(state: np.array, num_types: int) -> list:
    box_grid = state[0]
    agent_grid = state[2]
    if np.amax(agent_grid) != 255:
        return []
    return [calculate_item_type(box_grid, agent_grid, num_types)]


def drop_box(
    state: np.array,
    agent_item_type: list,
    ready_to_consume_types: list,
    outpoint_positions: list,
    entrypoint_positions: list,
    verbose=False,
) -> np.array:
    if all(item_type not in ready_to_consume_types for item_type in agent_item_type if len(ready_to_consume_types)):
        return (
            (deposit_item_in_grid(state, entrypoint_positions, outpoint_positions), "deposit item in grid")
            if verbose
            else deposit_item_in_grid(state, entrypoint_positions, outpoint_positions)
        )
    outpoint_position = get_nearest_outpoint(state[2], state[1], outpoint_positions, entrypoint_positions)
    return (deliver_box(outpoint_position), "deliver box") if verbose else deliver_box(outpoint_position)


def take_box(
    state: np.array,
    ready_to_consume_types: list,
    entrypoints_with_items: list,
    num_types: int,
    entrypoints,
    outpoints,
    verbose=False,
) -> np.array:
    if len(ready_boxes := get_ready_boxes_in_grid(state, ready_to_consume_types, num_types)):
        action = take_item_from_grid(state, ready_boxes, entrypoints, outpoints)
        return (action, "take item from grid") if verbose else action
    elif len(entrypoints_with_items):
        action = take_item_from_ep(entrypoints_with_items)
        return (action, "take item from ep") if verbose else action
    else:
        return (idle(), "idle") if verbose else idle()


def act(env: Storehouse, action: tuple, act_verbose: str = "") -> tuple:
    if action is not None:
        state, reward, done, info = env._step(action)
        render(env, act_verbose, action, reward, info)
    else:
        done = True
        reward = 0
        state = env.get_state()
        info = {}
    sleep(SLEEP_TIME)
    return state, reward, done, info


def get_nearest_outpoint(agent_grid: np.array, box_grid, outpoints: list, entrypoints):
    agent_position = get_max_position(agent_grid)[0]
    out_score = {pos: check_reachable(box_grid, agent_position, pos, entrypoints, outpoints) for pos in outpoints}
    return min(out_score, key=out_score.get)


def ehp(env: Storehouse, state: np.array, verbose=False):
    """
    EHP, fixed as a policy, usage:
        - To use for simulating the optimal policy, use the act function to move the
          agent and get newer states, and set as True the verbose flag
        - To use as a regular policy, just use it as a = p(env, s). The env variable
          is necessary to get some static info from the environment (positions mostly)
    """
    # Static information
    outpoint_positions = env.outpoints.outpoints
    num_types = len(env.type_information)
    entrypoint_positions = [ep.position for ep in env.entrypoints]
    ####
    ready_to_consume_types = get_ready_to_consume_items_from_state(state, outpoint_positions[0], num_types)
    entrypoints_with_items = get_ep_with_items_from_state(state, entrypoint_positions, num_types)
    agent_item_type = get_agent_item_type(state, num_types)
    if len(agent_item_type):
        return drop_box(
            state, agent_item_type, ready_to_consume_types, outpoint_positions, entrypoint_positions, verbose=verbose
        )
    else:
        return take_box(
            state,
            ready_to_consume_types,
            entrypoints_with_items,
            num_types,
            entrypoint_positions,
            outpoint_positions,
            verbose=verbose,
        )


@click.command()
@click.option("-l", "--log_folder", default=None)
@click.option("-p", "--policy", default="ehp")
@click.option("-c", "--conf_name", default="6x6fast")
@click.option("-m", "--max_steps", default=100)
@click.option("-v", "--visualize", default=0)
@click.option("-t", "--timesteps", default=STEPS)
@click.option("-se", "--save_episodes", default=False, type=int)
@click.option("-pc", "--path_cost", default=False)
@click.option("-r", "--random_Start", default=False)
@click.option("-w", "--path_reward_weight", default=0.5)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-mdp", "--mdp", default=False, type=bool)
def main(
    log_folder,
    policy,
    conf_name,
    max_steps,
    visualize,
    timesteps,
    save_episodes,
    path_cost,
    random_start,
    path_reward_weight,
    seed,
    mdp,
):
    global VISUAL
    global SLEEP_TIME
    VISUAL = int(visualize)
    SLEEP_TIME = 0.2 if visualize else 0.00
    env = Storehouse(
        log_folder or "log/log",
        logging=bool(log_folder),
        save_episodes=save_episodes,
        conf_name=conf_name,
        max_steps=int(max_steps),
        augment=False,
        transpose_state=True,
        path_cost=int(path_cost),
        random_start=int(random_start),
        path_reward_weight=path_reward_weight,
        seed=seed,
    )
    if mdp:
        data = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}
    s = env.reset(VISUAL)
    cum_reward = []
    cum_deliveries = []
    if not VISUAL:
        pbar = tqdm(total=timesteps)
    for _ in range(timesteps):
        if policy == "ehp_old":
            enhanced_human_policy(env, s)
        elif policy == "ihp":
            initial_human_policy(env, s)
        elif policy == "ehp":
            action, act_info = ehp(env, s, verbose=True)
            n_s, r, done, info = act(env, action, act_info)
            if mdp:
                data["state"].append(env.augment_state(s[0], s[1], s[2], AUGMENT_FACTOR))
                data["action"].append(env.denorm_action(action))
                data["reward"].append(r)
                data["next_state"].append(env.augment_state(n_s[0], n_s[1], n_s[2], AUGMENT_FACTOR))
                data["done"].append(done)
            s = n_s
            if done:
                cum_reward.append(env.current_return)
                cum_deliveries.append(env.score.delivered_boxes)
                s = env.reset()
                if VISUAL:
                    print("######################")
        else:
            raise NotImplementedError
        if not VISUAL:
            pbar.update(1)
    print(
        f"Finish! Results saved in {log_folder}.\nMean score: {np.mean(cum_reward)}. Delivered boxes: {np.mean(cum_deliveries)}"
    )
    if mdp:
        with open("mdp.pkl", "wb") as f:
            pkl.dump(data, f)


if __name__ == "__main__":
    main()
