import operator
from pathlib import Path
from time import sleep

import click
import numpy as np
from storehouse.environment.env_storehouse import CONF_NAME, MAX_MOVEMENTS, Storehouse
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
    return state


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
    return int_to_type(int(encoded_type / 255 * num_types))


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
    return boxes


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
    oldest_boxes_unsorted = np.where(grid == np.amax(grid))
    oldest_boxes = list(zip(oldest_boxes_unsorted[0], oldest_boxes_unsorted[1]))
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


def take_oldest_item(env, state, ready_boxes) -> np.array:
    age_grid = state[1]
    box_pos = get_oldest_box(age_grid, ready_boxes)
    state, reward, done, info = env._step(box_pos)
    return state, reward, done, info, box_pos


def deliver_box(env):
    action = env.outpoints.outpoints[0]
    state, reward, done, info = env._step(action)
    render(env, "deliver item", action, reward, info)
    s = check_reset(env, done)
    state = state if s is None else s
    sleep(SLEEP_TIME)
    return state


def take_item_from_ep(env: Storehouse, ep: list) -> None:
    da_box = ep[np.random.choice(len(ep))]
    _, reward, done, info = env._step(da_box)
    render(env, "deliver from entrypoint", da_box, reward, info)
    _ = check_reset(env, done)
    sleep(SLEEP_TIME)


def deliver_item_state(env: Storehouse, state: np.array, ready_boxes: list) -> np.array:
    _, reward, done, info, action = take_oldest_item(env, state, ready_boxes)
    render(env, "take item", action, reward, info)
    _ = check_reset(env, done)
    sleep(SLEEP_TIME)
    return deliver_box(env)


def get_direct_items(ep_items: dict, wanted_items: list) -> list:
    return [pos for pos, ep in ep_items.items() if len(wanted_items) and ep in wanted_items]


def deliver_from_ep_state(env: Storehouse, direct_items: list) -> np.array:
    take_item_from_ep(env, direct_items)
    return deliver_box(env)


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


def deposit_item_in_grid(env: Storehouse, state: np.array) -> np.array:
    box_grid = state[0]
    target_cell = find_target_cell(box_grid)
    if target_cell is None:
        done = True
    else:
        state, reward, done, info = env._step(target_cell)
        render(env, "store item in grid", target_cell, reward, info)
    s = check_reset(env, done)
    state = state if s is None else s
    sleep(SLEEP_TIME)
    return state


def store_item_state(env: Storehouse, state: np.array, ep_items: list) -> np.array:
    take_item_from_ep(env, ep_items)
    return deposit_item_in_grid(env, state)


def ehp_only_state(env: Storehouse, state: np.array):
    print(state)
    ready_to_consume_types = get_rtct_from_state(state, env.outpoints.outpoints[0], len(env.type_information))
    entrypoints_with_items = get_epwi_from_state(state, [ep.position for ep in env.entrypoints], len(env.type_information))
    ep_with_direct_items = get_direct_items(entrypoints_with_items, ready_to_consume_types)
    if len(ready_to_consume_types):
        if len(ready_boxes := get_ready_boxes_in_grid(state, ready_to_consume_types, len(env.type_information))):
            state = deliver_item_state(env, state, ready_boxes)
        elif len(ep_with_direct_items):
            state = deliver_from_ep_state(env, ep_with_direct_items)
        elif len(entrypoints_with_items):
            state = store_item_state(env, state, list(entrypoints_with_items.keys()))
        else:
            state = wait(env)
    elif len(entrypoints_with_items):
        state = store_item_state(env, state, list(entrypoints_with_items.keys()))
    else:
        state = wait(env)
    return state


@click.command()
@click.option("-l", "--log_folder", default="log/log")
@click.option("-p", "--policy", default="ehp")
@click.option("-c", "--conf_name", default="6x6fast")
@click.option("-m", "--max_steps", default=50)
@click.option("-r", "--render", default=0)
@click.option("-t", "--timesteps", default=STEPS)
@click.option("-s", "--save_episodes", default=False)
def main(log_folder, policy, conf_name, max_steps, render, timesteps, save_episodes):
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
    )

    s = env.reset(VISUAL)
    if not VISUAL:
        pbar = tqdm(total=timesteps)
    for _ in range(timesteps):
        if policy == "ehp":
            enhanced_human_policy(env, s)
        elif policy == "ihp":
            initial_human_policy(env, s)
        elif policy == "ehp_state":
            s = ehp_only_state(env, s)
        else:
            raise NotImplementedError
        if not VISUAL:
            pbar.update(1)
    print(f"Finish! Results saved in {log_folder}")
    # s, r, done, info = env.step(action)
    # env.render()
    # print(f'Action: {action}, Reward: {r}, info: {info}')
    # sleep(0.3)
    # if done:
    #     s = env.reset()


if __name__ == "__main__":
    main()
