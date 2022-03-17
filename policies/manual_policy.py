import operator
from time import sleep

import click
import numpy as np
from storehouse.environment.env_storehouse import Storehouse
from tqdm import tqdm

STEPS = 100000
SLEEP_TIME = 0.00
VISUAL = True


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
    da_box = np.random.choice(boxes) if len(boxes) > 1 else boxes[0]
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


def get_rtct_from_state(s: np.array, out_position: list, num_types: int) -> list:
    box_grid = s[0]
    outpoint_state = box_grid[out_position]
    if outpoint_state == 0:
        return []
    denormalized_out_state = denormalize_out_state(outpoint_state, num_types)
    wanted_material_encoded = deserialize_out_state(denormalized_out_state)
    return decode_material(wanted_material_encoded)


def denormalize_type(encoded_type: int, num_types: int) -> str:
    return int_to_type(int(encoded_type / 255 * num_types) - 1)


def get_epwi_from_state(s: np.array, ep_position: list, num_types: int) -> dict:
    box_grid = s[0]
    ep_state = {pos: box_grid[pos] for pos in ep_position}
    if not any(ep_state):
        return []
    return {pos: denormalize_type(ep, num_types) for pos, ep in ep_state.items() if ep}


def filter_restricted_boxes(grid: np.array, num_types: int) -> dict:
    adjacent_cells_delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    boxes = {}
    for row in range(1, grid.shape[0] - 1):
        for col in range(1, grid.shape[1] - 1):
            flag = True
            for delta in adjacent_cells_delta:
                adjacent_cell = tuple(map(operator.add, (row, col), delta))
                if grid[adjacent_cell] == 0:
                    flag = False
                    break
            if not flag and grid[row][col] > 0:  # Not restricted_cell and cell with item
                boxes[(row, col)] = denormalize_type(grid[row][col], num_types)
    return boxes  # form {pos: type}


def get_ready_boxes_in_grid(s: np.array, ready_types: set, num_types: int) -> list:
    box_grid = s[0]
    available_boxes = filter_restricted_boxes(box_grid, num_types)
    return [pos for pos, box in available_boxes.items() if box in ready_types]


def filter_age_grid(age_grid: np.array, ready_boxes: list) -> np.array:
    mask = np.zeros(age_grid.shape)
    for box in ready_boxes:
        mask[box] = 1
    return np.multiply(age_grid, mask)


def get_oldest_box(age_grid: np.array, ready_boxes: list) -> tuple:
    grid = filter_age_grid(age_grid, ready_boxes)
    oldest_boxes = get_max_position(grid)
    return oldest_boxes[np.random.choice(len(oldest_boxes))]


def render(env: Storehouse, general_action: str, action: tuple, reward: float, info: dict):
    if VISUAL:
        print(f"General action: {general_action}\nAction: {action}\nReward: {reward}\n{info}")
        env.render()


def check_reset(env, done) -> np.array:
    if done:
        if VISUAL:
            print("######################")
        return env.reset()
    return None


def take_item_from_grid(state, ready_boxes) -> np.array:
    age_grid = state[1]
    return get_oldest_box(age_grid, ready_boxes)


def deliver_box(outpoint_position):
    return outpoint_position


def take_item_from_ep(ep: list) -> None:
    return ep[np.random.choice(len(ep))]


def deposit_item_in_grid(state: np.array) -> np.array:
    box_grid = state[0]
    target_cell = find_target_cell(box_grid)
    return None if target_cell is None else target_cell


def idle():
    return (0, 1)


def find_target_cell(grid: np.array) -> tuple:
    target_cell = None
    for ii in range(1, grid.shape[0] - 1):
        for jj in range(1, grid.shape[1] - 1):
            if grid[ii][jj] == 0:
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
    outpoint_position: tuple,
    verbose=False,
) -> np.array:
    if any(item_type in ready_to_consume_types for item_type in agent_item_type if len(ready_to_consume_types)):
        return deliver_box(outpoint_position) if not verbose else (deliver_box(outpoint_position), "deliver box")
    else:
        return deposit_item_in_grid(state) if not verbose else (deposit_item_in_grid(state), "deposit item in grid")


def take_box(
    state: np.array, ready_to_consume_types: list, entrypoints_with_items: dict, num_types: int, verbose=False
) -> np.array:
    if len(ready_boxes := get_ready_boxes_in_grid(state, ready_to_consume_types, num_types)):
        action = take_item_from_grid(state, ready_boxes)
        return action if not verbose else (action, "take item from grid")
    elif len(entrypoints_with_items):
        action = take_item_from_ep(list(entrypoints_with_items.keys()))
        return action if not verbose else (action, "take item from ep")
    else:
        return idle() if not verbose else (idle(), "idle")


def act(env: Storehouse, action: tuple, act_verbose: str = "") -> tuple:
    if action is not None:
        state, reward, done, info = env._step(action)
    else:
        done = True
        reward = 0
    render(env, act_verbose, action, reward, info)
    s = check_reset(env, done)
    state = state if s is None else s
    sleep(SLEEP_TIME)
    return state, reward


def ehp_only_state(env: Storehouse, state: np.array, verbose=False):
    """
    EHP, fixed as a policy, usage:
        - To use for simulating the optimal policy, use the act function to move the
          agent and get newer states, and set as True the verbose flag
        - To use as a regular policy, just use it as a = p(env, s). The env variable
          is necessary to get some static info from the environment (positions mostly)
    """
    # Static information
    outpoint_position = env.outpoints.outpoints[0]
    num_types = len(env.type_information)
    entrypoint_position = [ep.position for ep in env.entrypoints]
    ####
    ready_to_consume_types = get_rtct_from_state(state, outpoint_position, num_types)
    entrypoints_with_items = get_epwi_from_state(state, entrypoint_position, num_types)
    agent_item_type = get_agent_item_type(state, num_types)
    if len(agent_item_type):
        return drop_box(state, agent_item_type, ready_to_consume_types, outpoint_position, verbose)
    else:
        return take_box(state, ready_to_consume_types, entrypoints_with_items, num_types, verbose)


@click.command()
@click.option("-l", "--log_folder", default=None)
@click.option("-p", "--policy", default="ehp")
@click.option("-c", "--conf_name", default="6x6fast")
@click.option("-m", "--max_steps", default=50)
@click.option("-r", "--render", default=0)
@click.option("-t", "--timesteps", default=STEPS)
@click.option("-s", "--save_episodes", default=False)
@click.option("-pc", "--path_cost", default=False)
def main(log_folder, policy, conf_name, max_steps, render, timesteps, save_episodes, path_cost):
    global VISUAL
    global SLEEP_TIME
    VISUAL = int(render)
    SLEEP_TIME = 0.2 if render else 0.00
    env = Storehouse(
        "log/log" if not log_folder else log_folder,
        logging=bool(log_folder),
        save_episodes=save_episodes,
        conf_name=conf_name,
        max_steps=int(max_steps),
        augment=False,
        transpose_state=True,
        path_cost=path_cost,
    )

    s = env.reset(VISUAL)
    cum_reward = 0
    if not VISUAL:
        pbar = tqdm(total=timesteps)
    for _ in range(timesteps):
        if policy == "ehp":
            enhanced_human_policy(env, s)
        elif policy == "ihp":
            initial_human_policy(env, s)
        elif policy == "ehp_state":
            action, act_info = ehp_only_state(env, s, verbose=True)
            s, r = act(env, action, act_info)
            cum_reward += r
        else:
            raise NotImplementedError
        if not VISUAL:
            pbar.update(1)
    print(f"Finish! Results saved in {log_folder}.\nMean score: {cum_reward / timesteps}")


if __name__ == "__main__":
    main()
