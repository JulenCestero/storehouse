import argparse
from pathlib import Path
from time import sleep

from parsecRL.environment.env_storehouse import Storehouse

SLEEP_TIME = 0.001
STEPS = 100000
log_folder = Path("log")


def random_agent(env=Storehouse(), timesteps: int = STEPS, render=True):
    env.reset(render)
    for _ in range(timesteps):
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
    args = parser.parse_args()
    logname = args.logname
    timesteps = args.timesteps
    main_folder = log_folder / logname
    main_folder.mkdir(parents=True, exist_ok=True)
    env = Storehouse(main_folder / logname, logging=True)
    random_agent(env, render=False, timesteps=timesteps)
