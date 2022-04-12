import itertools
import json
import logging
import operator
import os
import random
from math import atan2, ceil, prod
from pathlib import Path
from statistics import mean
from time import time
from typing import Iterator

import gym
import networkx as nx
import numpy as np
from colorama import Back, Fore, Style
from matplotlib import pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

CONF_NAME = "6x6"
MAX_ORDERS = 3
MAX_NUM_BOXES = 6
MIN_NUM_BOXES = 2
FEATURE_NUMBER = 3
MAX_INVALID = 10
MAX_MOVEMENTS = 1000  # 50
MIN_CNN_LEN = 32
MIN_SB3_SIZE = 32
EPISODE = 0
NUM_RANDOM_STATES = 500
PATH_REWARD_PROPORTION = 0.5


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

    def print_score(self) -> str:
        return (
            f"{self.delivered_boxes}, {self.filled_orders}, {self.clear_run_score}, {self.steps},"
            f"{self.ultra_negative_achieved}, {mean(self.box_ages)},"
            f"{self.non_optimal_material / max(1, self.delivered_boxes) * 100},"
            f"{self.timer}"
        )


class Box:
    def __init__(self, id: int, position: tuple, type: str = "A", age: int = 0):
        self.id = id
        self.type = type
        self.age = age
        self.position = position

    def update_age(self, num_steps: int = 1):
        self.age += max(num_steps, 1)

    def __eq__(self, other):
        if isinstance(other, Box):
            return self.position == other.position and self.age == other.age and self.id == other.id and self.age == other.age


class Agent:
    def __init__(self, initial_position: tuple, got_item: int = 0):
        # Only one object at time
        self.position = initial_position
        self.got_item = got_item  # Id of the box it is carrying, 0 if none

    def __eq__(self, other):
        if isinstance(other, Agent):
            return self.position == other.position and self.got_item == other.got_item


class Entrypoint:
    def __init__(self, position: tuple, type_information: dict):
        self.type_information = type_information
        self.position = position
        self.wait_time_cumulate = 0
        self.material_queue = []  # FIFO

    def create_new_order(self, order: dict, max_id: int):  # Order has the form {type: number_of_boxes}
        ((material_type, number_of_material),) = order.items()
        for _ in range(number_of_material):
            wait_time = round(np.random.poisson(self.type_information[material_type]["create"]["lambda"]))
            self.wait_time_cumulate += wait_time
            self.material_queue.append(
                {
                    "material": Box(max_id, self.position, material_type),
                    "timer": self.wait_time_cumulate,
                }
            )
            max_id += 1
        return max_id

    def get_item(self) -> Box:
        try:
            assert self.material_queue[0]["timer"] == 0
            return self.material_queue.pop(0)["material"]
        except Exception as ex:
            logging.error(f"Error at get_item in Entrypoint {self.position}")
            raise Exception from ex

    def update_entrypoint(self, steps: int = 1):
        for material in self.material_queue:
            material["timer"] = max(0, material["timer"] - max(1, steps))
        self.wait_time_cumulate = max(0, self.wait_time_cumulate - max(1, steps))
        try:
            assert self.material_queue[0]["timer"] == 0
            return self.material_queue[0]["material"].id
        except:
            return 0

    def reset(self):
        self.material_queue = []
        self.wait_time_cumulate = 0


class Outpoints:
    def __init__(self, outpoints: list, type_information: dict, delivery_timer: dict, max_orders: int):
        self.outpoints = outpoints  # Position
        self.type_information = type_information
        self.delivery_timer_info = delivery_timer
        self.max_orders = max_orders
        self.max_num_boxes = MAX_NUM_BOXES
        self.min_num_boxes = MIN_NUM_BOXES
        self.delivery_schedule = []  # Form: [{type, timer until ready, num_boxes}]
        self.desired_material = ""
        self.last_delivery_timers = np.Inf

    def reset(self):
        self.delivery_schedule = []
        self.desired_material = ""
        self.last_delivery_timers = np.Inf

    def update_timers(self, steps: int = 1):
        self.last_delivery_timers += max(1, steps)
        for delivery in self.delivery_schedule:
            delivery["timer"] = max(0, delivery["timer"] - max(1, steps))
        if len(self.delivery_schedule) > 0:
            if self.delivery_schedule[0]["timer"] == 0:
                self.desired_material = self.delivery_schedule[0]["type"]
            else:
                self.desired_material = ""
        else:
            self.desired_material = ""

    def create_order(self, type: str) -> dict:
        timer = round(np.random.poisson(self.type_information[type]["deliver"]["lambda"]))
        num_boxes = random.randrange(self.min_num_boxes, self.max_num_boxes + 1)
        return {"type": type, "timer": timer, "num_boxes": num_boxes}

    def create_delivery(self) -> dict:
        if self.last_delivery_timers <= np.random.poisson(self.delivery_timer_info["lambda"]):
            return None
        if len([order["timer"] for order in self.delivery_schedule if order["timer"] == 0]) > self.max_orders:
            return None
        box_type = random.choice(list(self.type_information.keys()))
        order = self.create_order(box_type)
        self.delivery_schedule.append(order)
        self.last_delivery_timers = 0
        return {key: item for key, item in order.items() if key != "timer"}

    def consume(self, box: Box) -> int:
        for ii, order in enumerate(self.delivery_schedule):
            if box.type == order["type"] and order["timer"] == 0:
                self.delivery_schedule[ii]["num_boxes"] -= 1
                if self.delivery_schedule[ii]["num_boxes"] < 1:
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
        max_orders: int = MAX_ORDERS,
        augment: bool = None,
        random_start: bool = False,
        normalized_state: bool = False,
        path_cost: bool = False,
        path_reward_weight: float = PATH_REWARD_PROPORTION,
    ):
        # logging.info(
        #     f"Logging: {logging}, save_episodes: {save_episodes}, max_steps: {max_steps}, conf_name: {conf_name}, augmented: {augment}, random_start: {random_start}, path_cost: {path_cost}, path_weights: {path_reward_weight}"
        # )
        self.signature = {}
        self.max_id = 1
        self.max_steps = max_steps
        self.max_orders = max_orders
        self.log_flag = logging
        self.path_reward_weight = path_reward_weight
        self.load_conf(conf_name)
        if augment is not None:
            self.augmented = augment
        self.random_start = random_start
        self.path_cost = path_cost
        self.normalized_state = normalized_state
        self.feature_number = FEATURE_NUMBER
        self.score = Score()
        self.episode = []
        self.available_actions = []
        self.invalid_actions = []
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
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], self.feature_number), dtype=np.uint8)
        self.material = {}  # dict of objects of the class box. {id: Box} form of the dict. ID > 0
        self.restricted_cells = []  # list of coordinates to where the agent cannot move
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
            self.random_initial_states = self.create_random_initial_states(NUM_RANDOM_STATES)
        if save_episodes:
            self.episode_folder = self.logname / "episodes"
            self.episode_folder.mkdir(parents=True, exist_ok=True)
        if self.log_flag:
            self.logname.mkdir(parents=True, exist_ok=True)
            self.metrics_log = f"{str(self.logname / self.logname.name)}_metrics.csv"
            with open(self.metrics_log, "a") as f:
                f.write("Delivered Boxes,Filled orders,Score,Steps,Ultra negative achieved,Mean box ages,Cueles,time\n")
            # self.actions_log = open(str(self.logname) + "_actions.csv", "w")
            # self.actions_log.write("")

    # def __del__(self):
    # if self.log_flag:
    # self.metrics_log.close()
    # self.actions_log.close()

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
            Entrypoint(position=eval(ii), type_information=self.type_information) for ii in conf["entrypoints"]
        ]
        self.outpoints = Outpoints(
            [eval(ii) for ii in conf["outpoints"]],
            type_information=self.type_information,
            delivery_timer=conf["delivery_timer"],
            max_orders=self.max_orders,
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
        prepared_matrix = np.array(matrix)
        prepared_matrix[start] = 0
        prepared_matrix[end] = 0
        for cell in whitelist:
            prepared_matrix[cell] = 0
        return Grid(matrix=np.negative(prepared_matrix) + 1)

    def find_path_cost(self, start_position, end_position) -> int:
        """
        Returns the cost of the movement. 0 if end_position is unreachable
        """
        grid = self.prepare_grid(self.grid, start_position, end_position, [ep.position for ep in self.entrypoints])
        start = grid.node(*reversed(start_position))
        end = grid.node(*reversed(end_position))
        path, runs = self.finder.find_path(start, end, grid)
        self.path = path
        return len(path) - 1  # If agent is idle, it counts as a movement of 1, we want 0

    def initialize_graph(self, shape: tuple):
        graph = nx.grid_2d_graph(*shape)
        graph.add_edges_from(
            [((x, y), (x + 1, y + 1)) for x in range(shape[0] - 1) for y in range(shape[1] - 1)]
            + [((x + 1, y), (x, y + 1)) for x in range(shape[0] - 1) for y in range(shape[1] - 1)]
        )
        self.floor_graph = graph

    @staticmethod
    def check_jump(ordered_cells: list) -> bool:
        no_duplicated_cells = list(dict.fromkeys(tuple(cell) for cell in ordered_cells))
        return any(
            abs(
                sum(
                    [
                        no_duplicated_cells[ii][0] - no_duplicated_cells[ii + 1][0],
                        no_duplicated_cells[ii][1] - no_duplicated_cells[ii + 1][1],
                    ]
                )
            )
            > 1
            for ii in range(-1, len(no_duplicated_cells) - 1)
        )

    @staticmethod
    def get_connected_cells(graph: nx.graph, shape: tuple, boxes: Iterator) -> Iterator:
        all_cells = set(itertools.product(range(shape[0]), range(shape[1])))
        cells_to_remove = all_cells - {box.position for box in boxes}
        graph.remove_nodes_from(cells_to_remove)
        return nx.connected_components(graph)

    @staticmethod
    def get_frontier(isle: list) -> list:
        """
        Returns a list of the cells of the borderline between the emtpy space and the box isles
        """
        cells = np.array([list(ii) for ii in isle])
        x_set = set(cells[:, 0])
        y_set = set(cells[:, 1])
        left = [cells[cells[:, 0] == x].min(axis=0) for x in x_set]  # left in render
        down = [cells[cells[:, 1] == y].max(axis=0) for y in y_set]  # left in render
        right = [cells[cells[:, 0] == x].max(axis=0) for x in x_set]  # left in render
        top = [cells[cells[:, 1] == y].min(axis=0) for y in y_set]  # left in render
        return left + right + top + down

    @staticmethod
    def order_points(cells: list) -> list:
        cent = sum(p[0] for p in cells) / len(cells), sum(p[1] for p in cells) / len(cells)
        cells.sort(key=lambda p: atan2(p[1] - cent[1], p[0] - cent[0]))
        return cells

    def calculate_restricted_cells(self):
        """
        !! BUGGY We need to find open donuts and also linear structures with a restricted cell in a corner
        Initial estimation of the restricted_cells list, where the agent is forbidden to navigate
        """
        self.restricted_cells = []
        return
        graph = np.array(self.floor_graph)
        isles = list(self.get_connected_cells(graph, self.grid.shape, self.material.values()))
        for isle in isles:
            frontier = self.get_frontier(isle)
            ordered_cells = self.order_points(frontier)
            if self.check_jump(ordered_cells):  # Buggy. Method works, but too restrictive.
                continue
            polygon = Polygon()
            self.restricted_cells.append(polygon)

    def __update_restricted_cells(self, ag: Agent):
        """DEPRECATED
        Updates the restricted_cells list where the agent is forbidden to navigate

        Args:
            ag (Agent): The agent that has changed the grid structure
        """
        adjacent_cells_delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if ag.got_item:
            for delta in adjacent_cells_delta:
                adjacent_cell = tuple(map(operator.add, ag.position, delta))
                if adjacent_cell in self.restricted_cells:
                    self.restricted_cells.remove(adjacent_cell)
        else:
            for delta in adjacent_cells_delta:
                adjacent_cell = tuple(map(operator.add, ag.position, delta))
                try:
                    assert self.grid[adjacent_cell] >= 0
                except:
                    continue
                flag = False
                if self.grid[adjacent_cell] > 0:
                    flag = True
                    for delta_delta in adjacent_cells_delta:
                        adjacent_adjacent_cell = tuple(map(operator.add, adjacent_cell, delta_delta))
                        try:
                            assert self.grid[adjacent_adjacent_cell] >= 0
                        except:
                            continue
                        if self.grid[adjacent_adjacent_cell] == 0:
                            flag = False
                            break
                if flag:
                    self.restricted_cells.append(adjacent_cell)

    @staticmethod
    def get_age_factor(age):
        """
        Age bounded within [0, 500]. Returns the percentage of the age factor [0, 1]. Linear (for now)
        """
        return min(max(age, 0), 500) / 500

    def get_entrypoints_with_items(self):
        try:
            entrypoints_with_items = [
                ep
                for ep in [ep for ep in self.entrypoints if len(ep.material_queue) > 0]
                if ep.material_queue[0]["timer"] == 0
            ]
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

    def get_reward(self, move_status: int, ag: Agent, box: Box = None) -> float:
        if move_status == 0:
            return -1
        elif move_status == 3:
            return -0.9
        macro_action_reward = self.get_macro_action_reward(ag, box)
        if macro_action_reward == -0.9:
            return -0.9
        weighted_reward = macro_action_reward
        if self.path_cost:
            w = self.path_reward_weight
            micro_action_reward = self.normalize_path_cost(len(self.path) - 1, self.grid.shape)
            weighted_reward = (1 - w) * macro_action_reward + w * micro_action_reward if micro_action_reward <= 0 else -1
        # assert weighted_reward <= 0
        return weighted_reward

    def delivery_reward(self, box):
        min_rew = -0.5
        oldest_box = max(
            (material for material in self.material.values() if material.type == box.type),
            key=operator.attrgetter("age"),
        )

        age_factor = self.get_age_factor(box.age)
        if box.id != oldest_box.id:
            self.score.non_optimal_material += 1
        self.score.delivered_boxes += 1
        return min_rew * age_factor

    def log(self, action):
        if self.done:
            self.score.box_ages += [box.age for box in self.material.values()]
            if not len(self.score.box_ages):
                self.score.box_ages.append(0)
            with open(self.metrics_log, "a") as f:
                f.write(self.score.print_score() + "\n")
            if self.save_episodes:
                with open(f"{self.episode_folder / self.logname.name}_episode_{EPISODE}.json", "w") as f:
                    json.dump(self.episode, f)
        # else:
        # self.actions_log.write(f"{action},{self.agents[0].got_item}\n")

    @staticmethod
    def normalize_age(age: int) -> float:
        return min(max(age, 0), 1000) / 1000 * 255

    def normalize_type(self, type: str) -> float:
        return (ord(type) - (ord("A") - 1)) * 255 / len(self.type_information)

    def decode_action(self, action: tuple):
        return action[0] * self.grid.shape[0] + action[1]

    def get_ready_to_consume_types(self):
        try:
            return {order["type"] for order in self.outpoints.delivery_schedule if order["timer"] == 0}
        except IndexError:
            return {}

    def get_available_actions(self) -> list:
        """
        BUGGY: Restricted cells logic broken.
        It's less restrictive than it should be
        """
        ### Assuming 1 agent
        agent = self.agents[0]
        ####################

        self.action_mask = np.zeros(self.action_mask.shape)
        ready_to_consume_types = self.get_ready_to_consume_types()

        if agent.got_item:  # Agent with item
            free_storage = []
            for ii in range(1, self.grid.shape[0] - 1):
                free_storage.extend(
                    (ii, jj)
                    for jj in range(1, self.grid.shape[1] - 1)
                    if self.grid[ii][jj] == 0 and (ii, jj) != agent.position
                )

            if self.material[agent.got_item].type in ready_to_consume_types:  # Outpoints open
                available_actions = list(self.outpoints.outpoints) + free_storage
            else:  # Outpoints closed
                available_actions = free_storage

        else:  # Agent without item
            entrypoints_with_items = self.get_entrypoints_with_items()
            if len(ready_to_consume_types) and len(entrypoints_with_items):  # If outpoints open, entrypoints open
                available_actions = [box.position for box in self.material.values() if box.type in ready_to_consume_types] + [
                    ep.position for ep in entrypoints_with_items
                ]
            elif len(ready_to_consume_types):  # If outpoints open, entrypoints closed
                if len([True for box in self.material.values() if box.type in ready_to_consume_types]):  # Item in storage
                    available_actions = [box.position for box in self.material.values() if box.type in ready_to_consume_types]
                else:  # Item NOT in storage
                    available_actions = self.get_available_cells()
            elif len(entrypoints_with_items):  # If outpoints closed, entrypoints open
                available_actions = [ep.position for ep in entrypoints_with_items]
            else:  # If outpoints closed, entrypoints closed
                available_actions = self.get_available_cells()
        try:
            for action in available_actions:
                self.action_mask[self.decode_action(action)] = 1
        except IndexError as e:
            print(e)
            print(available_actions)
            raise IndexError from e
        self.available_actions = available_actions
        return available_actions

    def check_cell_in_restricted_cell(self, point: tuple) -> bool:
        return any(p.contains(Point(*point)) for p in self.restricted_cells)

    def get_free_storage_of_grid(self):
        result = []
        for ii in range(1, self.grid.shape[0] - 1):
            result.extend(
                (
                    (ii, jj)
                    for jj in range(1, self.grid.shape[1] - 1)
                    # if self.grid[ii][jj] == 0 and (ii, jj) not in self.restricted_cells
                    if self.grid[ii][jj] == 0 and not self.check_cell_in_restricted_cell((ii, jj))
                )
            )
        return result

    def get_available_cells(self):
        free_storage = self.get_free_storage_of_grid()
        outer_ring = []
        try:
            assert self.grid.shape[0] == self.grid.shape[1]  # Only for square environments
        except AssertionError as e:
            raise NotImplementedError from e
        return self.get_outer_ring_with_free_storage(outer_ring, free_storage)

    def get_outer_ring_with_free_storage(self, outer_ring, free_storage):
        for pos in range(self.grid.shape[0] - 1):
            outer_ring.extend(
                (
                    (0, pos),
                    (self.grid.shape[0] - 1, pos),
                    (pos, 0),
                    (pos, self.grid.shape[0] - 1),
                )
            )

        outer_ring = [
            action
            for action in set(outer_ring)
            if action not in [entrypoint.position for entrypoint in self.entrypoints] + self.outpoints.outpoints
        ]

        return free_storage + outer_ring

    def set_signature(self, signature: dict) -> None:
        self.reset(force_clean=True)
        self.done = signature["done"]
        self.agents = [Agent(agent["pos"], agent["item_id"]) for agent in signature["agents"]]
        self.material = {box["id"]: Box(box["id"], box["pos"], box["type"], box["age"]) for box in signature["boxes"]}
        self.calculate_restricted_cells()
        self.outpoints.delivery_schedule = [el.copy() for el in signature["outpoints"]["delivery_schedule"]]
        self.outpoints.desired_material = signature["outpoints"]["desired_material"]
        self.outpoints.last_delivery_timers = signature["outpoints"]["last_delivery_timers"]
        for ep, info in zip(self.entrypoints, signature["entrypoints"]):
            ep.material_queue = [
                {
                    "timer": int(el["timer"]),
                    "material": Box(el["material"].id, el["material"].position, el["material"].type),
                }
                for el in info["material_queue"]
            ]
            ep.wait_time_cumulate = info["wait_time_cumulate"]
            ep.position = info["pos"]
            # ep.material_queue = [{"timer": item["timer"], "type": item["material"].type} for item in
            #           copy.deepcopy(info["queue"])]
        self.num_actions = signature["num_actions"]
        for box_id, box in list(self.material.items()) + [
            (queue[0]["material"].id, queue[0]["material"])
            for queue in [ep.material_queue for ep in self.entrypoints if len(ep.material_queue) > 0]
            if queue[0]["timer"] == 0
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
            "boxes": [
                {
                    "id": id_box,
                    "pos": box.position,
                    "age": box.age,
                    "type": box.type,
                }
                for id_box, box in self.material.items()
            ],
            "restricted_cell": list(self.restricted_cells),
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
                    "material_queue": [
                        {
                            "timer": int(el["timer"]),
                            "material": Box(el["material"].id, el["material"].position, el["material"].type),
                        }
                        for el in ep.material_queue
                    ],
                    "wait_time_cumulate": ep.wait_time_cumulate,
                }
                for ep in self.entrypoints
            ],
            "outpoints": {
                "pos": list(self.outpoints.outpoints),
                "accepted_types": list(set(self.get_ready_to_consume_types())),
                "delivery_schedule": [el.copy() for el in self.outpoints.delivery_schedule],
                "desired_material": self.outpoints.desired_material,
                "last_delivery_timers": self.outpoints.last_delivery_timers,
            },
            "num_actions": self.num_actions,
        }

    def save_state_simplified(self, reward: int, action: tuple):
        state = self.get_signature()
        self.episode.append(
            {
                "step": action,
                "reward": reward,
                "cum_reward": self.current_return,
                "path": self.path,
                "state": {
                    key: value
                    if key not in ["entrypoints", "outpoints"]
                    else {"pos": value["pos"], "accepted_types": value["accepted_types"]}
                    if key == "outpoints"
                    else {"pos": [pos["pos"] for pos in value]}
                    for key, value in state.items()
                    if key not in ["material", "agents", "done", "max_id"]
                },
            }
        )

    def augment_state(self, box_grid, age_grid, agent_grid) -> np.array:
        return np.array(
            [
                np.kron(grid, np.ones((self.augment_factor, self.augment_factor)))
                for grid in np.array([box_grid, age_grid, agent_grid])
            ]
        )

    def mix_state(self, box_grid, age_grid, agent_grid):
        return (
            self.augment_state(box_grid, age_grid, agent_grid)
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

    def normalize_type_combination(self, ready_to_consume_types: list, num_types: int) -> float:
        num = sum([2 ** self.type_to_int(consume_type) for consume_type in ready_to_consume_types] + [0])
        return num * 255 / (2 ** num_types - 1)

    def construct_age_grid(self, age_grid):
        for box in self.material.values():
            # box_grid[box.position] = self.normalize_type(box.type)
            age_grid[box.position] = self.normalize_age(box.age)
        return age_grid

    def construct_box_grid(self, box_grid):
        for box in list(self.material.values()) + [
            ep.material_queue[0]["material"] for ep in self.get_entrypoints_with_items()
        ]:
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

    def __construct_av_action_grid(self, available_action_grid):
        for action in self.get_available_actions():
            available_action_grid[action] = 255
        return available_action_grid

    def initialize_grids(self):
        return np.zeros(self.grid.shape), np.zeros(self.grid.shape), np.zeros(self.grid.shape)

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
        return state_mix if self.transpose_state else state_mix.reshape(size + (self.feature_number,))
        # state_mix = state_mix.reshape(size + (self.feature_number,))
        # return state_mix.transpose([2, 0, 1]) if self.transpose_state else state_mix

    def assert_movement(self, ag: Agent, movement: tuple) -> int:
        """
        TODO: simplify this mess
        """
        try:  # Checking if the new position is valid
            _ = self.grid[movement]
            assert all(ii >= 0 for ii in movement)
            # assert ag.position != movement # Stay still
            if ag.got_item:
                assert movement not in [entrypoint.position for entrypoint in self.entrypoints]
                assert not (
                    movement in self.outpoints.outpoints
                    and self.material[ag.got_item].type
                    not in [material["type"] for material in self.outpoints.delivery_schedule if material["timer"] < 1]
                )  #  Go to not ready outpoints
                assert movement not in [box.position for box in self.material.values()]
                assert not (
                    movement not in [entrypoint.position for entrypoint in self.entrypoints]
                    and movement not in self.outpoints.outpoints
                    and (movement[0] in (0, self.grid.shape[0] - 1) or movement[1] in (0, self.grid.shape[1] - 1))
                )  # Move outer ring
                # assert not (movement in self.outpoints.outpoints and ag.got_item != self.outpoints.desired_material) # Incorrect material
            else:
                assert movement not in self.outpoints.outpoints
                # assert not (movement in [entrypoint.position for entrypoint in self.entrypoints] and not any([entrypoint.material_queue[0]['timer'] for entrypoint in self.entrypoints])) # Move to entrypoints if there is no material
            assert self.find_path_cost(ag.position, movement) >= 0
        except (AssertionError, IndexError):
            # logging.warning('Invalid movement')
            self.num_invalid += 1
            return 0
        return 1

    def __assert_movement(self, movement: tuple) -> int:
        """
        IDEAL WORLD...
        """
        if movement in self.available_actions:
            return 1
        self.num_invalid += 1
        return 0

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
        self.grid[ag.position] = ag.got_item
        self.material[ag.got_item].position = movement
        ag.got_item = 0
        self.calculate_restricted_cells()
        # self.update_restricted_cells(ag)  # Update the restricted cell list with the new actions
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
        self.calculate_restricted_cells()
        # self.update_restricted_cells(ag)  # Update the restricted cell list with the new actions
        return 1

    # def check_full_occupation(self):
    #     max_occupation = 0.9  # Magic number
    #     return len(self.material) >= (self.grid.shape[0] - 2) * (self.grid.shape[1] - 2) * max_occupation

    def _step(self, action: tuple, render=False) -> list:
        self.last_action = self.denorm_action(action)
        self.num_actions += 1
        info = {"Steps": self.num_actions}
        agent = self.agents[0]  # Assuming 1 agent
        # Done conditions
        if not len(self.available_actions):  # If storehouse full
            self.score.ultra_negative_achieved = True
            self.done = True
        if self.num_actions >= self.max_steps:
            self.done = True
            reward = 0
            info["done"] = "Max movements achieved. Well done!"
            if self.log_flag:
                self.log(action)
            return self.return_result(reward, info)
        ####
        self.score.steps += 1
        if self.log_flag:
            self.log(action)
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
        if order is not None:
            self.max_id = random.choice(self.entrypoints).create_new_order(
                {order["type"]: order["num_boxes"]}, self.max_id
            )  # TODO: Create load balancer?
        if self.save_episodes:
            self.save_state_simplified(reward, action)
        if render:
            self.render()
        return self.return_result(reward, info)

    def get_idle_time(self):
        timers = [order["timer"] for order in self.outpoints.delivery_schedule] + [
            ep.material_queue[0]["timer"] for ep in self.entrypoints if len(ep.material_queue) > 0
        ]
        try:
            return min(timers)
        except ValueError as ex:
            return 1

    def detect_idle(self) -> bool:
        A = bool(self.agents[0].got_item)
        O = bool(self.outpoints.desired_material)
        S = bool(len(self.material))
        E = any(not bool(ep.material_queue[0]["timer"]) for ep in self.entrypoints if len(ep.material_queue) > 0)
        return (not A and not E and not O) or (not A and not E and not S)

    def update_timers(self):
        idle_time = self.get_idle_time()
        steps = len(self.path) - 1 if self.path_cost else 1
        if self.detect_idle():
            steps = max(steps, idle_time)
        self.score.timer += steps
        for box in self.material.values():
            box.update_age(steps)
        self.outpoints.update_timers(steps)
        for entrypoint in self.entrypoints:
            self.grid[entrypoint.position] = entrypoint.update_entrypoint(steps)

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
        if move_status == 0:
            self.invalid_actions.append(action)
        else:
            self.invalid_actions = []
        result = self.get_reward(move_status, agent, box)
        self.score.clear_run_score += result
        return result, move_status

    def return_result(self, reward, info):
        self.last_r = reward
        self.current_return += reward
        info["timer"] = self.score.timer
        info["delivered"] = self.score.delivered_boxes
        info["entrypoint queue"] = [len(entrypoint.material_queue) for entrypoint in self.entrypoints]
        info["outpoint queue"] = self.outpoints.delivery_schedule
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
            type=type or random.choice(list(self.type_information.keys())),
            age=age or random.choice(range(1, 100)),
        )

        self.max_id += 1
        return box

    def assign_order_to_material(self):
        def decomposition(i):
            while i > 0:
                try:
                    n = random.randint(MIN_NUM_BOXES, min(i, MAX_NUM_BOXES))
                except ValueError:
                    n = random.randint(1, min(i, MAX_NUM_BOXES))
                yield n
                i -= n

        for type in self.type_information.keys():
            num_boxes_type = len([box for box in self.material.values() if box.type == type])
            num_boxes_distribution = decomposition(num_boxes_type)
            for num_boxes in num_boxes_distribution:
                timer = round(np.random.poisson(10))  # Magic number
                self.outpoints.delivery_schedule.append({"type": type, "timer": timer, "num_boxes": num_boxes})

    def create_random_initial_states(self, num_states) -> list:
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
        random_flag = self.random_start
        self.signature = {}
        self.restricted_cells = []
        self.invalid_actions = []
        self.episode = []
        self.grid = np.zeros(self.grid.shape)
        self.num_actions = 0
        self.current_return = 0
        self.material = {}
        self.outpoints.reset()
        for entrypoint in self.entrypoints:
            entrypoint.reset()

        if random_flag and not force_clean:
            self.set_signature(np.random.choice(self.random_initial_states))
        else:
            self.agents = [Agent(initial_position=(3, 3)) for _ in range(self.num_agents)]
        self.initialize_graph(self.grid.shape)
        self.calculate_restricted_cells()
        self.done = False
        self.score.reset()
        self.num_invalid = 0
        self.number_actions = 0
        self.current_return = 0
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
                # (random.choice(range(1, self.grid.shape[0] - 1)), random.choice(range(1, self.grid.shape[1] - 1))),
                (0, 1),  # To ensure that the agent doesn't start trapped
                got_item=random.choice([0, self.max_id]),  # If the agent has an item, it will be of ID = 1
            )
            for _ in range(self.num_agents)
        ]
        if self.agents[0].got_item:  # Initialize random box
            self.material[self.agents[0].got_item] = self.create_random_box(position=self.agents[0].position)
        for row, col in itertools.product(range(1, self.grid.shape[0] - 1), range(1, self.grid.shape[1] - 1)):
            if (row, col) == self.agents[0].position:
                continue
            if random.random() < box_probability:
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
        return " " if num == 0 else chr(int(ord("A") - 1 + num))
        # return str(num)

    @staticmethod
    def decode(letter: str) -> int:
        """
        Decodes the grid from letters to numbers
        """
        return ord(letter) - (ord("A") - 1)

    def render_state(self, dark=True):

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
                    encoded_el = [
                        self.entrypoints[ii].material_queue[0]["material"].type
                        for ii in range(len(self.entrypoints))
                        if self.entrypoints[ii].position == (r, e)
                    ][0]
                else:
                    encoded_el = self.material[element].type
                try:
                    if element > 0 and self.check_cell_in_restricted_cell((r, e)):
                        encoded_el = f"{Back.RED}{Fore.BLACK}{self.material[element].type}{Style.RESET_ALL}"
                    for agent in self.agents:
                        if agent.position == (r, e):
                            if agent.got_item:
                                encoded_el = f"{Back.BLUE}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                            else:
                                encoded_el = f"{Back.CYAN}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                    if (r, e) in [entrypoint.position for entrypoint in self.entrypoints]:
                        encoded_el = f"{Back.GREEN}{Fore.BLACK}{encoded_el}{Style.RESET_ALL}"
                    if (r, e) in self.outpoints.outpoints:
                        if self.outpoints.desired_material:
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


if __name__ == "__main__":
    from time import sleep

    env = Storehouse(random_start=True)
    n_a = env.action_space.n
    for _ in range(10):
        env.reset(1)
        # env.render()
        done = False
        t = 0
        while not done and t < 100:
            a = np.random.choice(n_a)
            s, r, done, inf = env.step(a)
            print(f"Action: {env.norm_action(a)}, Reward: {r}")
            env.render()
            t += 1
            sleep(0.5)
