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
    if len(boxes) > 1:
        da_box = np.random.choice(boxes)
    else:
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
        [
            material
            for material in env.material.values()
            if material.type in available_types and material.position not in env.restricted_cells
        ],
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
    state, reward, done, info = env._step((0, 1))
    if VISUAL:
        print(f"General action: wait\nAction: {(0,1)}\nReward: {reward}\n{info}")
        env.render()
    if done:
        if VISUAL:
            print("######################")
        env.reset()
    sleep(SLEEP_TIME)


def initial_human_policy(env: Storehouse, state: np.array):
    try:
        ready_to_consume_types = [order["type"] for order in env.outpoints.delivery_schedule if order["timer"] == 0]
    except IndexError:
        ready_to_consume_types = list()
    try:
        entrypoints_with_items = [
            ep for ep in [ep for ep in env.entrypoints if len(ep.material_queue) > 0] if ep.material_queue[0]["timer"] == 0
        ]
    except IndexError:
        entrypoints_with_items = list()
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
        ready_to_consume_types = list()
    try:
        entrypoints_with_items = [
            ep for ep in [ep for ep in env.entrypoints if len(ep.material_queue) > 0] if ep.material_queue[0]["timer"] == 0
        ]
    except IndexError:
        entrypoints_with_items = list()
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


@click.command()
@click.argument("log_folder")
@click.option("-p", "--policy", default="ehp")
@click.option("-c", "--conf_name", default="6x6fast")
@click.option("-m", "--max_steps", default=50)
@click.option("-r", "--render", default=0)
def main(log_folder, policy, conf_name, max_steps, render):
    global VISUAL
    global SLEEP_TIME
    VISUAL = int(render)
    if render:
        SLEEP_TIME = 0.2
    else:
        SLEEP_TIME = 0.00
    folder = Path(log_folder)
    folder.mkdir(parents=True, exist_ok=True)
    env = Storehouse(log_folder, logging=True, save_episodes=False, conf_name=conf_name, max_steps=max_steps)
    s = env.reset(VISUAL)
    if not VISUAL:
        pbar = tqdm(total=STEPS)
    for _ in range(STEPS):
        if policy == "ehp":
            enhanced_human_policy(env, s)
        elif policy == "ihp":
            initial_human_policy(env, s)
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
