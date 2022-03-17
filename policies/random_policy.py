from pathlib import Path
from time import sleep

import click
from storehouse.environment.env_storehouse import CONF_NAME, MAX_MOVEMENTS, Storehouse
from tqdm import tqdm

SLEEP_TIME = 0.00
STEPS = 100000
log_folder = Path("log")


def random_agent(env=Storehouse(), timesteps: int = STEPS, render=True):
    env.reset(render)
    cum_reward = 0
    for _ in tqdm(range(timesteps)):
        action = env.action_space.sample()
        s, r, done, info = env.step(action)
        cum_reward += r
        if render:
            env.render()
            print(f"Action: {action}\nReward: {r}\n{info}")
            sleep(SLEEP_TIME)
        if done:
            s = env.reset(render)
    return cum_reward / timesteps


@click.command()
@click.option("-l", "--log_folder", default="log/log")
@click.option("-c", "--conf_name", default="6x6fast")
@click.option("-m", "--max_steps", default=50)
@click.option("-r", "--render", default=0)
@click.option("-pc", "--path_cost", default=False)
@click.option("-t", "--timesteps", default=STEPS)
def main(log_folder, conf_name, max_steps, render, path_cost, timesteps):
    env = Storehouse(
        "log/log" if not log_folder else log_folder,
        logging=bool(log_folder),
        save_episodes=False,
        conf_name=conf_name,
        max_steps=int(max_steps),
        path_cost=path_cost,
        augment=False,
    )

    mean_reward = random_agent(env, timesteps=timesteps, render=render)
    print(f"Results saved in {log_folder}. Mean reward: {mean_reward}")


if __name__ == "__main__":
    main()
