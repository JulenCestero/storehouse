import itertools
import json
import logging
import operator
import os
from math import atan2, ceil, prod
from pathlib import Path
from statistics import mean
from time import time
from typing import Iterator

import gym
import networkx as nx
import numpy as np
from colorama import Back, Fore, Style
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.stats import poisson
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage.morphology import flood_fill

CONF_NAME = "6x6fast"
MAX_NUM_BOXES = 6
MIN_NUM_BOXES = 2
FEATURE_NUMBER = 3
MAX_INVALID = 10
MAX_MOVEMENTS = 100  # 50
MIN_CNN_LEN = 32
MIN_SB3_SIZE = 32
EPISODE = 0
NUM_RANDOM_STATES = 2000
PATH_REWARD_PROPORTION = 0.0
TYPE_CODIFICATION = {"A": 100, "B": 200}
TYPE_COMB_CODIFICATION = {0: 0, 1: 50, 2: 100, 3: 150, 4: 200, 5: 255}


class Score:
    def __init__(self):
        self.reset()

    def reset(self):
        self.delivered_boxes = 0
        self.filled_orders = 0
        self.clear_run_score = 0
        self.ultra_negative_achieved = False
        self.steps = 0
        self.box_ages = []
        self.non_optimal_material = 0
        self.timer = 0
        self.max_id = 0
        self.total_orders = 0
        self.seed = 0
        self.returns = []
        self.discounted_return = 0

    def print_score(self) -> str:
        return (
            f"{self.delivered_boxes}, {self.filled_orders}, {self.clear_run_score}, {self.steps},"
            f"{self.ultra_negative_achieved}, {mean(self.box_ages)},"
            f"{self.non_optimal_material / max(1, self.delivered_boxes) * 100},"
            f"{self.timer},"
            f"{self.max_id},"
            f"{self.total_orders},"
            f"{self.seed},"
            f"{self.discounted_return}"
        )


class Box:
    def __init__(self, id: int, position: tuple, type: str = "A", age: int = 1):
        self.id = id
        self.type = type
        self.age = age
        self.position = position

    def update_age(self, num_steps: int = 1):
        self.age += max(num_steps, 1)

    def __eq__(self, other):
        if isinstance(other, Box):
            return self.position == other.position and self.age == other.age and self.id == other.id and self.age == other.age

    def __repr__(self) -> str:
        return f"Box(id={self.id}, type={self.type}, age={self.age})"


class Agent:
    def __init__(self, initial_position: tuple, got_item: int = 0):
        # Only one object at time
        self.position = initial_position
        self.got_item = got_item  # Id of the box it is carrying, 0 if none

    def __eq__(self, other):
        if isinstance(other, Agent):
            return self.position == other.position and self.got_item == other.got_item


class Entrypoint:
    def __init__(self, position: tuple, type_information: dict, rng: list):
        self.rng = rng
        self.type_information = type_information
        self.position = position
        self.material_queue = []  # FIFO

    def create_new_material(self, max_id: int):
        box_type = self.rng[0].choice(list(self.type_information.keys()))
        prob = self.type_information[box_type]["create"]
        if self.rng[0].choice([True, False], p=[prob, 1 - prob]):
            material = Box(max_id, self.position, box_type)
            self.material_queue.append(material)
            max_id += 1
        return max_id

    def get_item(self) -> Box:  # sourcery skip: raise-specific-error
        try:
            return self.material_queue.pop(0)
        except IndexError as ex:
            logging.error(f"Error at get_item in Entrypoint {self.position}")
            raise IndexError from ex

    def update_entrypoint(self, max_id, steps: int = 1):
        for material in self.material_queue:
            material.update_age(max(1, steps))
        return self.create_new_material(max_id)

    def reset(self):
        self.material_queue = []


class Delivery:
    def __init__(
        self,
        prob: int,
        num_boxes: int,
        type: str,
        rng: list,
        timer: int = 0,
        ready: bool = False,
    ):
        self.type = type
        self.prob = prob
        self.num_boxes = num_boxes
        self.timer = timer
        self.ready = ready
        self.rng = rng

    def update_timer(self, step: int = 1):
        self.timer += step
        if not self.ready:
            # prob = poisson.cdf(self.timer, self.prob)
            prob = self.prob
            if self.rng[0].choice([True, False], p=[prob, 1 - prob]):
                self.ready = True

    def __repr__(self) -> str:
        return (
            f"Delivery(type={self.type}, num_boxes={self.num_boxes}, timer={self.timer}, ready={self.ready}, prob={self.prob})"
        )

    def __eq__(self, other):
        if isinstance(other, Delivery):
            return (
                self.type == other.type
                and self.prob == other.prob
                and self.num_boxes == other.num_boxes
                and self.timer == other.timer
                and self.ready == other.ready
            )


class Outpoints:
    def __init__(self, outpoints: list, type_information: dict, delivery_prob: dict, rng: list):
        self.outpoints = outpoints  # Position
        self.type_information = type_information
        self.delivery_prob = delivery_prob
        self.max_num_boxes = MAX_NUM_BOXES
        self.min_num_boxes = MIN_NUM_BOXES
        self.delivery_schedule = []  # Form: [Delivery()]
        self.last_delivery_timers = np.Inf
        self.rng = rng

    def reset(self):
        self.delivery_schedule = []
        self.last_delivery_timers = np.Inf

    def update_timers(self, steps: int = 1):
        self.last_delivery_timers += max(1, steps)
        for delivery in self.delivery_schedule:
            delivery.update_timer(max(0, max(1, steps)))

    def create_order(self, type: str) -> dict:
        num_boxes = self.rng[0].integers(self.min_num_boxes, self.max_num_boxes + 1)
        return Delivery(type=type, prob=self.type_information[type]["deliver"], num_boxes=num_boxes, rng=self.rng)
        # return {"type": type, "timer": timer, "num_boxes": num_boxes}

    def create_delivery(self) -> dict:
        prob = self.delivery_prob
        if not self.rng[0].choice([True, False], p=[prob, 1 - prob]):
            return None
        box_type = self.rng[0].choice(list(self.type_information.keys()))
        order = self.create_order(box_type)
        self.delivery_schedule.append(order)
        self.last_delivery_timers = 0
        return order

    def consume(self, box: Box) -> int:
        for ii, order in enumerate(self.delivery_schedule):
            if box.type == order.type and order.ready:
                self.delivery_schedule[ii].num_boxes -= 1
                if self.delivery_schedule[ii].num_boxes < 1:
                    del self.delivery_schedule[ii]
                    return 2
                return 1
        return 0


class Storehouse(gym.Env):
    def __init__(
        self,
        logname: str = "log/log",
        logging: bool = False,
        save_episodes: bool = False,
        transpose_state: bool = False,
        max_steps: int = MAX_MOVEMENTS,
        conf_name: str = CONF_NAME,
        augment: bool = None,
        random_start: bool = False,
        normalized_state: bool = False,
        path_reward_weight: float = PATH_REWARD_PROPORTION,
        seed: int = None,  # used for random_start
        reward_function: int = 0,  # To choose between reward functions
        gamma: float = 0.99,
    ):
        # logging.info(
        #     f"Logging: {logging}, save_episodes: {save_episodes}, max_steps: {max_steps}, conf_name: {conf_name}, augmented: {augment}, random_start: {random_start}, path_cost: {path_cost}, path_weights: {path_reward_weight}"
        # )
        if reward_function == 0:
            self.get_reward = self.get_reward
        elif reward_function == 1:
            self.get_reward = self.__get_reward_1
        self.signature = {}
        self.rng = [np.random.default_rng(seed)]
        self.max_id = 1
        self.gamma = gamma
        self.max_steps = max_steps
        self.log_flag = logging
        self.path_reward_weight = path_reward_weight
        self.load_conf(conf_name)
        if augment is not None:
            self.augmented = augment
        self.random_start = random_start
        self.normalized_state = normalized_state
        self.feature_number = FEATURE_NUMBER
        self.score = Score()
        self.original_seed = seed
        self.episode = []
        self.available_actions = []
        self.logname = Path(logname)
        self.save_episodes = save_episodes
        self.transpose_state = transpose_state
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        if self.augmented:
            self.augment_factor = ceil(MIN_SB3_SIZE / min(self.grid.shape))
            size = tuple(dimension * self.augment_factor for dimension in self.grid.shape)
        else:
            size = self.grid.shape
        self.action_space = gym.spaces.Discrete(self.grid.shape[0] * self.grid.shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(size[0], size[1], self.feature_number), dtype=np.uint8
        )
        self.material = {}  # dict of objects of the class box. {id: Box} form of the dict. ID > 0
        self.agents = [Agent((0, 0)) for _ in range(self.num_agents)]
        self.done = False
        self.action = None
        self.floor_graph = None
        self.path = []
        self.num_actions = 0
        self.num_invalid = 0
        self.current_return = 0
        self.action_mask = np.zeros(len(list(range(self.action_space.n))))
        if self.random_start:
            self.random_initial_states = self.create_random_initial_states(NUM_RANDOM_STATES, seed)
        if save_episodes:
            self.episode_folder = self.logname / "episodes"
            self.episode_folder.mkdir(parents=True, exist_ok=True)
        if self.log_flag:
            self.logname.mkdir(parents=True, exist_ok=True)
            self.metrics_log = f"{str(self.logname / self.logname.name)}_metrics.csv"
            with open(self.metrics_log, "a") as f:
                f.write(
                    "Delivered Boxes,Filled orders,Score,Steps,Ultra negative achieved,Mean box ages,FIFO violation,time,max_id,Total orders,Seed,Discounted return\n"
                )

    def load_conf(self, conf: str = CONF_NAME):
        """
        Load the configuration from a JSON file .

        Args:
            conf (str, optional): Configuration name to be loaded. Defaults to CONF_NAME.
        """
        with open(os.path.join(os.path.dirname(__file__), "conf.json"), "r") as f:
            current_conf = json.load(f)[conf]
        self.augmented = any(length < MIN_CNN_LEN for length in current_conf["grid"])
        self.grid = np.zeros(current_conf["grid"])
        conf = current_conf["conf"]
        self.type_information = conf["material_types"]
        self.entrypoints = [
            Entrypoint(position=eval(ii), type_information=self.type_information, rng=self.rng) for ii in conf["entrypoints"]
        ]
        self.outpoints = Outpoints(
            [eval(ii) for ii in conf["outpoints"]],
            type_information=self.type_information,
            delivery_prob=conf["delivery_prob"],
            rng=self.rng,
        )
        self.num_agents = conf["num_agents"]

    def outpoints_consume(self):  # sourcery skip: raise-specific-error
        """
        Function that consumes all the materials dropped in outpoints. If the material is not listed
        within the self.material list, an error is raised
        """
        for outpoint in self.outpoints.outpoints:
            if self.grid[outpoint] > 0:
                try:
                    status = self.outpoints.consume(self.material[self.grid[outpoint]])
                    if status == 1:
                        logging.info("Material consumed")
                    elif status == 2:
                        logging.info("Order completed")
                        self.score.filled_orders += 1
                    self.score.box_ages.append(self.material[self.grid[outpoint]].age)
                    del self.material[self.grid[outpoint]]
                    self.grid[outpoint] = 0
                except Exception as e:
                    logging.error(f"Unexpected error at consuming the material at outpoint {outpoint}: {e}")
                    raise Exception from e

    @staticmethod
    def prepare_grid(matrix: np.array, start: tuple, end: tuple, whitelist: list = None) -> Grid:
        prepared_matrix = np.array(matrix, dtype="int16")
        prepared_matrix[start] = 0
        prepared_matrix[end] = 0
        for cell in whitelist:
            prepared_matrix[cell] = 0
        return Grid(matrix=np.negative(prepared_matrix) + 1)

    def find_path_cost(self, start_position, end_position) -> int:
        """
        Returns the cost of the movement. 0 if end_position is unreachable
        """
        grid = self.prepare_grid(
            self.grid,
            start_position,
            end_position,
            whitelist=[ep.position for ep in self.entrypoints] + self.outpoints.outpoints,
        )
        start = grid.node(*reversed(start_position))
        end = grid.node(*reversed(end_position))
        path, runs = self.finder.find_path(start, end, grid)
        self.path = path
        return len(path) - 1  # If agent is idle, it counts as a movement of 1, we want 0

    @staticmethod
    def __get_age_factor(age):
        """
        Age bounded within [0, 500]. Returns the percentage of the age factor [0, 1]. Linear (for now)
        [UPDATE/FIX] Added '1 - ...' to give more weight (therefore, worse rw) to newer items, instead of older items
        """
        bound = 500
        bounded_age = min(max(abs(age), 0), bound) / bound
        return (1 - bounded_age) ** 2 + (bounded_age) ** 2

    @staticmethod
    def get_age_factor(age, old_age):
        """
        Age bounded within [0, 500]. Returns the percentage of the age factor [0, 1]. Linear (for now)
        [UPDATE/FIX] Added '1 - ...' to give more weight (therefore, worse rw) to newer items, instead of older items
        """
        bound = 500
        return min(max(abs(age - old_age), 0), bound) / bound

    def delivery_reward(self, box):
        min_rew = -0.5
        oldest_box = max(
            [material for material in self.material.values() if material.type == box.type]
            + [
                ep.material_queue[0]
                for ep in self.entrypoints
                if ep.material_queue
                if ep.material_queue[0].type == box.type and self.path[0] != ep.position
            ],
            key=operator.attrgetter("age"),
        )
        age_factor = self.get_age_factor(box.age, oldest_box.age)
        self.score.delivered_boxes += 1
        if box.id != oldest_box.id:
            self.score.non_optimal_material += 1
        return min_rew * age_factor

    def get_entrypoints_with_items(self):
        try:
            entrypoints_with_items = [ep for ep in self.entrypoints if len(ep.material_queue) > 0]
        except IndexError:
            entrypoints_with_items = []
        return entrypoints_with_items

    def get_macro_action_reward(self, ag: Agent, box: Box = None) -> float:
        if self.get_ready_to_consume_types():
            if ag.position in self.outpoints.outpoints:
                return self.delivery_reward(box)
            return -0.9 if len(self.material) or self.get_entrypoints_with_items() else 0
        elif not self.get_ready_to_consume_types() and self.get_entrypoints_with_items():
            return -0.9
        else:
            return 0

    @staticmethod
    def normalize_path_cost(cost: int, grid_shape: tuple) -> float:
        return -cost / (prod(grid_shape) / 2)

    def __new_reward(self):
        """
        [-0.9, 0] ~ in practice [-0.5, 0]
        """
        return max(
            -0.9,
            (
                -sum(
                    [box.age for box in self.material.values()]
                    + [box.age for sublist in [ep.material_queue for ep in self.entrypoints] for box in sublist]
                )
                / 10000
            ),
        )

    def __get_reward_2(self, move_status: int, ag: Agent, box: Box = None) -> float:
        new_r = self.__new_reward()
        if move_status == 0:
            return -0.5 + new_r
        elif move_status == 2 and ag.position in self.outpoints.outpoints:
            self.score.delivered_boxes += 1
            return 0
            # return new_r
        else:
            return new_r

    def __get_reward_1(self, move_status: int, ag: Agent, box: Box = None) -> float:
        if move_status == 2 and ag.position in self.outpoints.outpoints:
            self.score.delivered_boxes += 1
            return 1
        return -1

    def get_reward(self, move_status: int, ag: Agent, box: Box = None) -> float:
        if move_status == 0:
            return -1
        macro_action_reward = self.get_macro_action_reward(ag, box)
        if macro_action_reward == -0.9:
            return -0.9
        weighted_reward = macro_action_reward
        w = self.path_reward_weight
        micro_action_reward = self.normalize_path_cost(len(self.path) - 1, self.grid.shape)
        weighted_reward = (1 - w) * macro_action_reward + w * micro_action_reward if micro_action_reward <= 0 else -1
        return weighted_reward

    def log(self):
        self.score.box_ages += [box.age for box in self.material.values()]
        self.score.max_id = self.max_id
        self.score.seed = self.original_seed
        self.score.discounted_return = np.mean(
            [
                sum(self.gamma**ii * ret for ii, ret in enumerate(self.score.returns[jj:]))
                for jj in range(len(self.score.returns))
            ]
        )
        if not len(self.score.box_ages):
            self.score.box_ages.append(0)
        with open(self.metrics_log, "a") as f:
            f.write(self.score.print_score() + "\n")
        if self.save_episodes:
            with open(f"{self.episode_folder / self.logname.name}_episode_{EPISODE}.json", "w") as f:
                json.dump(self.episode, f)

    @staticmethod
    def normalize_age(age: int) -> float:
        return ceil(min(max(age, 0), 1000) / 1000 * 255)

    def normalize_type(self, type: str) -> int:
        # return (ord(type) - (ord("A") - 1)) * 255 / len(self.type_information)
        return TYPE_CODIFICATION[type]

    def get_ready_to_consume_types(self) -> dict:
        try:
            return {order.type for order in self.outpoints.delivery_schedule if order.ready}
        except IndexError:
            return {}

    def check_reachable(self, position: tuple, maze: np.array) -> bool:
        adjacent_cells_delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for delta in adjacent_cells_delta:
            adjacent_cell = tuple(map(operator.add, position, delta))
            try:
                if maze[adjacent_cell] == 0.5:
                    return True
            except IndexError:
                continue
        return False

    def get_available_actions(self) -> list:
        # Get reachable cells
        agent = self.agents[0]
        maze = np.array(self.grid)
        for ep in self.entrypoints:
            maze[ep.position] = 0
        for op in self.outpoints.outpoints:
            maze[op] = 0
        maze[agent.position] = 0
        maze = flood_fill(maze, agent.position, 0.5, connectivity=1)
        if agent.got_item:
            for ep in self.entrypoints:
                maze[ep.position] = 0
            if self.material[agent.got_item].type not in self.get_ready_to_consume_types():
                for op in self.outpoints.outpoints:
                    maze[op] = 0
        else:
            available_boxes = [pos for pos in np.argwhere(self.grid > 0) if self.check_reachable(pos, maze)]
            for pos in available_boxes:
                maze[tuple(pos)] = 0.5
            for op in self.outpoints.outpoints:
                maze[op] = 0
        self.action_mask = maze.reshape(-1) == 0.5
        return self.action_mask

    def set_signature(self, signature: dict) -> None:
        self.reset(force_clean=True)
        self.done = signature["done"]
        # self.rng = [signature["rng"][0]]
        self.agents = [Agent(agent["pos"], agent["item_id"]) for agent in signature["agents"]]
        self.material = {box["id"]: Box(box["id"], box["pos"], box["type"], box["age"]) for box in signature["boxes"]}
        self.outpoints.delivery_schedule = [
            Delivery(type=el.type, prob=el.prob, num_boxes=el.num_boxes, timer=el.timer, ready=el.ready, rng=self.rng)
            for el in signature["outpoints"]["delivery_schedule"]
        ]
        self.outpoints.last_delivery_timers = signature["outpoints"]["last_delivery_timers"]
        for ep, info in zip(self.entrypoints, signature["entrypoints"]):
            ep.material_queue = [Box(el.id, el.position, el.type, el.age) for el in info["material_queue"]]
            ep.position = info["pos"]
        self.num_actions = signature["num_actions"]
        for box_id, box in list(self.material.items()) + [
            (queue[0].id, queue[0]) for queue in [ep.material_queue for ep in self.entrypoints if len(ep.material_queue) > 0]
        ]:
            self.grid[box.position] = box_id
        if self.agents[0].got_item:
            self.grid[self.agents[0].position] = 0
        self.max_id = signature["max_id"]
        self.signature = signature

    def get_signature(self) -> dict:
        return {
            "max_id": self.max_id,
            "done": self.done,
            "rng": [self.rng[0]],
            "boxes": [
                {
                    "id": id_box,
                    "pos": box.position,
                    "age": box.age,
                    "type": box.type,
                }
                for id_box, box in self.material.items()
            ],
            "agents": [
                {
                    "pos": agent.position,
                    "item": self.material[agent.got_item].type if agent.got_item > 0 else 0,
                    "item_id": int(agent.got_item) if agent.got_item > 0 else 0,
                }
                for agent in self.agents
            ],
            "entrypoints": [
                {
                    "pos": ep.position,
                    "material_queue": [Box(el.id, el.position, el.type, el.age) for el in ep.material_queue],
                }
                for ep in self.entrypoints
            ],
            "outpoints": {
                "pos": list(self.outpoints.outpoints),
                "accepted_types": list(set(self.get_ready_to_consume_types())),
                "delivery_schedule": [
                    Delivery(type=el.type, prob=el.prob, num_boxes=el.num_boxes, timer=el.timer, ready=el.ready, rng=self.rng)
                    for el in self.outpoints.delivery_schedule
                ],
                "last_delivery_timers": self.outpoints.last_delivery_timers,
            },
            "num_actions": self.num_actions,
        }

    def save_state_simplified(self, reward: int, action: tuple):
        state = self.get_signature()
        self.episode.append(
            {
                "step": action,
                "num_actions": state["num_actions"],
                "reward": reward,
                "cum_reward": self.current_return,
                "path": self.path,
                "state": {
                    "agents": state["agents"],
                    "outpoints": {"pos": state["outpoints"]["pos"], "accepted_types": state["outpoints"]["accepted_types"]},
                    "entrypoints": {"pos": [ep["pos"] for ep in state["entrypoints"]]},
                    "boxes": state["boxes"],
                },
            }
        )

    @staticmethod
    def augment_state(box_grid, age_grid, agent_grid, augment_factor) -> np.array:
        return np.array(
            [np.kron(grid, np.ones((augment_factor, augment_factor))) for grid in np.array([box_grid, age_grid, agent_grid])]
        )

    def mix_state(self, box_grid, age_grid, agent_grid):
        return (
            self.augment_state(box_grid, age_grid, agent_grid, self.augment_factor)
            if self.augmented
            else np.array([box_grid, age_grid, agent_grid])
        )

    @staticmethod
    def normalize_state(state_mix):
        for ii, matrix in enumerate(state_mix):
            state_mix[ii] = matrix / 255
        return state_mix

    @staticmethod
    def type_to_int(box_type: str) -> int:
        """
        Converts A, B, C... to 0, 1, 2,...
        """
        return ord(box_type) - ord("A")

    def normalize_type_combination(self, ready_to_consume_types: list, num_types: int) -> int:
        num = sum([2 ** self.type_to_int(consume_type) for consume_type in ready_to_consume_types] + [0])
        # return num * 255 / (2**num_types - 1)
        return TYPE_COMB_CODIFICATION[num]

    def construct_age_grid(self, age_grid):
        for box in self.material.values():
            # box_grid[box.position] = self.normalize_type(box.type)
            age_grid[box.position] = self.normalize_age(box.age)
        for ep in self.entrypoints:
            if ep.material_queue:
                oldest_box = ep.material_queue[0]
                age_grid[ep.position] = self.normalize_age(oldest_box.age)
        return age_grid

    def construct_box_grid(self, box_grid):
        for box in list(self.material.values()) + [ep.material_queue[0] for ep in self.get_entrypoints_with_items()]:
            box_grid[box.position] = self.normalize_type(box.type)
        for agent in self.agents:
            if agent.position in [ep.position for ep in self.entrypoints] and agent.got_item:
                box_grid[agent.position] = self.normalize_type(self.material[agent.got_item].type)
        ready_to_consume_types = self.get_ready_to_consume_types()
        for pos in self.outpoints.outpoints:
            box_grid[pos] = self.normalize_type_combination(ready_to_consume_types, len(self.type_information))
        return box_grid

    def construct_agent_grid(self, agent_grid):
        for agent in self.agents:
            agent_grid[agent.position] = 255 if agent.got_item else 128
        return agent_grid

    def initialize_grids(self):
        return (
            np.zeros(self.grid.shape, dtype=np.uint8),
            np.zeros(self.grid.shape, dtype=np.uint8),
            np.zeros(self.grid.shape, dtype=np.uint8),
        )

    def construct_grids(self):
        box_grid, age_grid, agent_grid = self.initialize_grids()
        return self.construct_box_grid(box_grid), self.construct_age_grid(age_grid), self.construct_agent_grid(agent_grid)

    def get_state(self) -> list:
        box_grid, age_grid, agent_grid = self.construct_grids()
        state_mix = self.mix_state(box_grid, age_grid, agent_grid)
        size = state_mix[0].shape
        if self.normalized_state:
            state_mix = self.normalize_state(state_mix)
        self.signature = self.get_signature()
        return (state_mix if self.transpose_state else state_mix.reshape(size + (self.feature_number,))).astype("uint8")

    def assert_movement(self, ag: Agent, movement: tuple) -> int:
        try:  # Checking if the new position is valid
            _ = self.grid[movement]
            assert all(ii >= 0 for ii in movement)
            assert self.get_available_actions()[self.denorm_action(movement)]
            assert self.find_path_cost(ag.position, movement) >= 0
        except (AssertionError, IndexError):
            self.num_invalid += 1
            return 0
        return 1

    def move_agent(self, ag: Agent, movement: tuple) -> int:
        """Move an agent to a new position .

        Args:
            ag (Agent): Agent to perform the movement
            movement (tuple): New cell coordinates to interact with

        Returns:
            int:    0 for invalid action.
                    1 for correct take
                    2 for correct drop
                    3 for non optimal action (move to empty cell without object)

        !IMPROVEMENT IDEA: Delete all the asserts and just check if the movement is in the available movements
        """
        if not self.assert_movement(ag, movement):
            return 0
        if self.grid[movement] > 0:  # If cell has object TAKE
            return self.take_item(ag, movement)
        else:  # If cell doesnt have object DROP
            return self.drop_item(ag, movement)

    def drop_item(self, ag, movement):
        try:
            assert ag.got_item  # Check if it can drop an object
        except AssertionError:
            ag.position = movement
            return 3
        ag.position = movement
        if (
            movement not in [entrypoint.position for entrypoint in self.entrypoints]
            and movement not in self.outpoints.outpoints
            and (movement[0] in (0, self.grid.shape[0] - 1) or movement[1] in (0, self.grid.shape[1] - 1))
        ):  # Move to outern ring
            self.material[ag.got_item].position = movement
            return 3
        self.grid[ag.position] = ag.got_item
        self.material[ag.got_item].position = movement
        ag.got_item = 0
        return 2

    def take_item(self, ag, movement):
        try:
            assert not ag.got_item  # Check if it can take an object
            if movement not in [entrypoint.position for entrypoint in self.entrypoints]:
                assert self.grid[movement] in [box.id for box in self.material.values()]
        except AssertionError:
            # logging.error("Agent has object but ordered to take another object")
            return 0
        ag.position = movement
        ag.got_item = self.grid[ag.position]  # Store ID of the taken object
        self.grid[ag.position] = 0
        return 1

    def _step(self, action: tuple, render=False) -> list:
        action = (int(action[0]), int(action[1]))
        self.last_action = self.denorm_action(action)
        self.num_actions += 1
        info = {"Steps": self.num_actions}
        agent = self.agents[0]  # Assuming 1 agent
        if self.num_actions >= self.max_steps:
            self.done = True
            reward = 0
            info["done"] = "Max movements achieved. Well done!"
            if self.log_flag:
                self.log(action)
            return self.return_result(reward, info)
        ####
        self.score.steps += 1
        # Update environment with the agent interaction
        if not self.done:
            reward, move_status = self.act(agent, action, info)
        else:
            info["Info"] = "Done. Please reset the environment"
            reward = -1e3
            return self.return_result(reward, info)
        # Update environment unrelated to agent interaction
        self.outpoints_consume()
        self.update_timers()
        order = self.outpoints.create_delivery()
        if order is not None and self.log_flag:
            self.score.total_orders += 1
        if self.save_episodes:
            self.save_state_simplified(reward, action)
        if render:
            self.render()
        return self.return_result(reward, info)

    def update_timers(self):
        steps = len(self.path) - 1
        self.score.timer += steps
        for box in self.material.values():
            box.update_age(steps)
        self.outpoints.update_timers(steps)
        for entrypoint in self.entrypoints:
            self.max_id = entrypoint.update_entrypoint(max_id=self.max_id, steps=steps)
            try:
                self.grid[entrypoint.position] = entrypoint.material_queue[0].id
            except IndexError as ex:
                self.grid[entrypoint.position] = 0

    def act(self, agent, action, info):
        # Movement
        start_cell = agent.position
        move_status = self.move_agent(agent, action)
        if move_status in (1, 2):  # If interacted with a Box
            if move_status == 1 and agent.position in [
                entrypoint.position for entrypoint in self.entrypoints
            ]:  # Added new box into the system
                box = [entrypoint for entrypoint in self.entrypoints if entrypoint.position == agent.position][0].get_item()
                self.material[box.id] = box
            else:
                box = self.material[self.grid[agent.position] if self.grid[agent.position] > 0 else agent.got_item]
            info["Info"] = f"Box {box.id} moved"
        else:
            box = None
        reward = self.get_reward(move_status, agent, box)
        self.score.clear_run_score += reward
        self.score.returns.append(reward)
        return reward, move_status

    def return_result(self, reward, info):
        self.last_r = reward
        self.current_return += reward
        info["timer"] = self.score.timer
        info["delivered"] = self.score.delivered_boxes
        info["outpoint queue"] = {
            t: sum((deliver.num_boxes) for deliver in self.outpoints.delivery_schedule if deliver.type == t and deliver.ready)
            for t in self.get_ready_to_consume_types()
        }
        for entrypoint in self.entrypoints:
            info[f"EP{entrypoint.position}"] = list(entrypoint.material_queue)

        self.last_info = info
        return self.get_state(), reward, self.done, info

    def norm_action(self, action) -> tuple:
        assert action < self.grid.shape[0] * self.grid.shape[1] and action >= 0
        return (int(action / self.grid.shape[0]), int(action % self.grid.shape[0]))

    def denorm_action(self, action: tuple) -> int:
        return action[0] * self.grid.shape[0] + action[1]

    def step(self, action: int) -> list:
        self.action = self.norm_action(action)
        assert action == self.denorm_action(self.action)
        state, reward, done, info = self._step(self.action)
        return state, reward, done, info

    def create_random_box(self, position: tuple, type: str = None, age: int = None):
        box = Box(
            id=self.max_id,
            position=position,
            type=type or self.rng[0].choice(list(self.type_information.keys())),
            age=age or self.rng[0].choice(range(1, 100)),
        )

        self.max_id += 1
        return box

    def assign_order_to_material(self):
        def decomposition(i):
            while i > 0:
                try:
                    n = self.rng[0].integers(MIN_NUM_BOXES, min(i, MAX_NUM_BOXES) + 1)
                except ValueError:
                    n = self.rng[0].integers(1, min(i, MAX_NUM_BOXES) + 1)
                yield n
                i -= n

        for type, info in self.type_information.items():
            num_boxes_type = len([box for box in self.material.values() if box.type == type])
            num_boxes_distribution = decomposition(num_boxes_type)
            for num_boxes in num_boxes_distribution:
                self.outpoints.delivery_schedule.append(
                    Delivery(type=type, prob=info["deliver"], num_boxes=num_boxes, rng=self.rng)
                )

    def create_random_initial_states(self, num_states, seed=None) -> list:
        self.rng[0] = np.random.default_rng(seed)
        self.original_seed = seed
        states = []
        print("Creating random states...")
        t0 = time()
        for _ in range(num_states):
            self.reset_random()
            states.append(self.get_signature())
        print(f"Finished! Created {num_states} states in {time() - t0}s")
        return states

    def set_search(self):
        self.log_flag = False
        self.save_episodes = False

    def reset(self, render=False, force_clean=False) -> list:
        global EPISODE
        EPISODE += 1
        self.max_id = 1
        random_flag = self.random_start
        self.signature = {}
        self.episode = []
        self.grid = np.zeros(self.grid.shape)
        self.num_actions = 0
        self.current_return = 0
        self.material = {}
        self.last_action = 0
        self.last_r = 0
        self.last_info = {}
        self.outpoints.reset()
        for entrypoint in self.entrypoints:
            entrypoint.reset()

        if random_flag and not force_clean:
            self.set_signature(self.rng[0].choice(self.random_initial_states))
        else:
            self.agents = [Agent(initial_position=(3, 3)) for _ in range(self.num_agents)]
        self.done = False
        self.score.reset()
        self.num_invalid = 0
        self.number_actions = 0
        if not len(self.get_available_actions()):
            return self.reset(render)
        if render:
            self.render()
        return self.get_state()

    def reset_random(self):
        self.reset(force_clean=True)
        box_probability = 0.4  # Magic number
        self.agents = [
            Agent(
                (0, 1),  # To ensure that the agent doesn't start trapped
                got_item=self.rng[0].choice([0, self.max_id]),
            )
            for _ in range(self.num_agents)
        ]
        if self.agents[0].got_item:  # Initialize random box
            self.material[self.agents[0].got_item] = self.create_random_box(position=self.agents[0].position)
        for row, col in itertools.product(range(1, self.grid.shape[0] - 1), range(1, self.grid.shape[1] - 1)):
            if (row, col) == self.agents[0].position:
                continue
            if self.rng[0].random() < box_probability:
                max_id = self.max_id
                self.material[max_id] = self.create_random_box((row, col))
        for box in list(self.material.values()):  # Introduce these boxes in the grid object
            self.grid[box.position] = box.id
        if self.agents[0].got_item:  # Clean grid
            self.grid[self.agents[0].position] = 0
        self.assign_order_to_material()

    @staticmethod
    def encode(num: int) -> str:
        """
        Encodes the grid from numbers to letters
        """
        # return " " if num == 0 else chr(int(ord("A") - 1 + num))
        return " " if num == 0 else [key for key, value in TYPE_CODIFICATION.items() if num == value][0]

    @staticmethod
    def decode(letter: str) -> int:
        """
        Decodes the grid from letters to numbers
        """
        # return ord(letter) - (ord("A") - 1)
        return TYPE_CODIFICATION[letter]

    def render_state(self, dark=True):
        from matplotlib import pyplot as plt

        state = (
            np.flip(np.rot90(np.transpose(self.get_state().reshape((self.feature_number,) + self.grid.shape)), k=3), axis=1)
            / 255.0
        )
        if not dark:
            state = abs(state - 1)
        plt.clf()
        plt.imshow(state)
        plt.draw()
        plt.pause(10e-10)

    def render(self):
        """
        COLORMAP GUIDE:

        Green: empty entrypoint
        Yellow: entrypoint with item
        Cyan: agent emptyhanded
        Blue: agent with item
        Purple: outpoint
        Red: Restricted cells
        Letter with black background: Box of the letter type

        (I hope you like spaghetti)
        """
        maze = "" + "+"
        for _ in range(self.grid.shape[1] * 2 - 1):
            maze += "-"
        maze += "+\n"
        for r, row in enumerate(self.grid):
            maze += "|"
            for e, element in enumerate(row):
                if element == 0:
                    encoded_el = " "
                elif (r, e) in [entrypoint.position for entrypoint in self.entrypoints]:
                    encoded_el = [ep.material_queue[0].type for ep in self.entrypoints if ep.position == (r, e)][0]
                else:
                    encoded_el = self.material[element].type
                try:
                    # if element > 0 and self.check_cell_in_restricted_cell((r, e)):
                    # encoded_el = f"{Back.RED}{Fore.BLACK}{self.material[element].type}{Style.RESET_ALL}"
                    for agent in self.agents:
                        if agent.position == (r, e):
                            if agent.got_item:
                                encoded_el = f"{Back.BLUE}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                            else:
                                encoded_el = f"{Back.CYAN}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                    if (r, e) in [entrypoint.position for entrypoint in self.entrypoints]:
                        encoded_el = f"{Back.GREEN}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                    if (r, e) in self.outpoints.outpoints:
                        if self.get_ready_to_consume_types():
                            encoded_el = f"{Back.MAGENTA}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                        else:
                            encoded_el = f"{Back.WHITE}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                except:
                    pass
                maze += encoded_el
                if e < self.grid.shape[1] - 1:
                    maze += ":"
            maze += "|\n"
        maze += "+"
        for _ in range(self.grid.shape[1] * 2 - 1):
            maze += "-"
        maze += "+\n"
        print(maze)

    def seed(self, seed: int = ...) -> list:
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        self.rng[0] = np.random.default_rng(seed)
        self.original_seed = seed
        return [seed]


if __name__ == "__main__":
    from time import sleep

    env = Storehouse(logging=True, random_start=True, save_episodes=False, max_steps=100)
    n_a = env.action_space.n
    for _ in range(10):
        env.reset(1)
        # env.render()
        done = False
        t = 0
        while not done and t < 105:
            a = np.random.choice(n_a)
            s, r, done, inf = env.step(a)
            # print(f"Action: {env.norm_action(a)}, Reward: {r}, Info: {inf}")
            # env.render()
            t += 1
            # sleep(0.5)
