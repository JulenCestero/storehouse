import operator
from time import sleep

import numpy as np
from storehouse.environment.env_storehouse import Storehouse

STEPS = 100000
SLEEP_TIME = 0.2
VISUAL = True


def store_item(env: Storehouse, waiting_items: list):
    target_entrypoint = [ep.position for ep in waiting_items][0]
    state, reward, done, info = env._step(target_entrypoint)
    if VISUAL:
        print(f"General action: store_item\nAction: {target_entrypoint}\nReward: {reward}\n{info}")
        env.render()
    if done:
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
        print("######################")
        env.reset()
    sleep(SLEEP_TIME)

    state, reward, done, info = env._step(env.outpoints.outpoints[0])
    if VISUAL:
        print(f"General action: deliver_item\nAction: {env.outpoints.outpoints[0]}\nReward: {reward}\n{info}")
        env.render()
    if done:
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
        print("######################")
        env.reset()
    sleep(SLEEP_TIME)

    state, reward, done, info = env._step(env.outpoints.outpoints[0])
    if VISUAL:
        print(f"General action: deliver_item\nAction: {env.outpoints.outpoints[0]}\nReward: {reward}\n{info}")
        env.render()
    if done:
        print("######################")
        env.reset()
    sleep(SLEEP_TIME)


def wait(env: Storehouse):
    state, reward, done, info = env._step((0, 1))
    if VISUAL:
        print(f"General action: wait\nAction: {(0,1)}\nReward: {reward}\n{info}")
        env.render()
    if done:
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


def main():
    env = Storehouse("PRUEBA/manual_policy", logging=True, save_episodes=False, conf_name="6x6fast", max_steps=100)
    s = env.reset(VISUAL)
    for _ in range(STEPS):
        # enhanced_human_policy(env, s)
        initial_human_policy(env, s)
        # s, r, done, info = env.step(action)
        # env.render()
        # print(f'Action: {action}, Reward: {r}, info: {info}')
        # sleep(0.3)
        # if done:
        #     s = env.reset()


if __name__ == "__main__":
    main()
