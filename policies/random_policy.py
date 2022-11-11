from pathlib import Path
from time import sleep

import click
from storehouse.environment.env_storehouse import CONF_NAME, MAX_MOVEMENTS, Storehouse
from tqdm import tqdm

SLEEP_TIME = 0.00
STEPS = 100000
log_folder = Path("log")


def random_agent(env=Storehouse(), timesteps: int = STEPS, visualize=True):
    env.reset(visualize)
    cum_reward = 0
    for _ in tqdm(range(timesteps)):
        action = env.action_space.sample()
        s, r, done, info = env.step(action)
        cum_reward += r
        if visualize:
            print(f"Action: {action}\nReward: {r}\n{info}")
            env.render()
            sleep(SLEEP_TIME)
        if done:
            s = env.reset(visualize)
    return cum_reward / timesteps


def run_random_train(
    log_folder,
    conf_name,
    max_steps,
    visualize,
    save_episodes,
    timesteps,
    random_start,
    path_reward_weight,
    seed,
    reward,
    gamma,
):
    global SLEEP_TIME
    SLEEP_TIME = 0.2 if visualize else SLEEP_TIME
    env = Storehouse(
        log_folder or "log/log",
        logging=bool(log_folder),
        save_episodes=save_episodes,
        conf_name=conf_name,
        max_steps=int(max_steps),
        augment=False,
        random_start=random_start,
        path_reward_weight=path_reward_weight,
        seed=seed,
        reward_function=reward,
        gamma=gamma,
    )
    env.seed(seed)
    mean_reward = random_agent(env, timesteps=timesteps, visualize=visualize)
    print(f"Results saved in {log_folder}. Mean reward: {mean_reward * int(max_steps)}")


@click.command()
@click.option("-l", "--log_folder", default=None)
@click.option("-c", "--conf_name", default="6x6fast")
@click.option("-m", "--max_steps", default=100)
@click.option("-v", "--visualize", default=0)
@click.option("-r", "--random_start", default=0)
@click.option("-se", "--save_episodes", default=False, type=int)
@click.option("-t", "--timesteps", default=STEPS)
@click.option("-w", "--path_reward_weight", default=0.0)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-rw", "--reward", default=0, type=int)
@click.option("-g", "--gamma", default=0.99, type=float)
def main(
    log_folder,
    conf_name,
    max_steps,
    visualize,
    save_episodes,
    timesteps,
    random_start,
    path_reward_weight,
    seed,
    reward,
    gamma,
):
    run_random_train(
        log_folder,
        conf_name,
        max_steps,
        visualize,
        save_episodes,
        timesteps,
        random_start,
        path_reward_weight,
        seed,
        reward,
        gamma,
    )


if __name__ == "__main__":
    main()
