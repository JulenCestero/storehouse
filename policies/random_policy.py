import argparse
from pathlib import Path
from time import sleep

from storehouse.environment.env_storehouse import CONF_NAME, MAX_MOVEMENTS, Storehouse
from tqdm import tqdm

SLEEP_TIME = 0.00
STEPS = 100000
log_folder = Path("log")


def random_agent(env=Storehouse(), timesteps: int = STEPS, render=True):
    env.reset(render)
    for _ in tqdm(range(timesteps)):
        action = env.action_space.sample()
        s, r, done, info = env.step(action)
        if render:
            env.render()
            print(f"Action: {action}\nReward: {r}\n{info}")
            sleep(SLEEP_TIME)
        if done:
            s = env.reset(render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logname", help="Name of the folder where the results of the training will be stored")
    parser.add_argument("-t", "--timesteps", help="Number of timesteps used for the training", default=STEPS, type=int)
    parser.add_argument("-c", "--conf_name", default=CONF_NAME)
    parser.add_argument("-m", "--max_steps", default=MAX_MOVEMENTS)
    args = parser.parse_args()
    logname = args.logname
    timesteps = args.timesteps
    conf_name = args.conf_name
    max_steps = int(args.max_steps)
    main_folder = log_folder / logname
    main_folder.mkdir(parents=True, exist_ok=True)
    env = Storehouse(main_folder / logname, logging=True, conf_name=conf_name, max_steps=max_steps)
    random_agent(env, render=False, timesteps=timesteps)
    print(f"Results saved in {logname}")
