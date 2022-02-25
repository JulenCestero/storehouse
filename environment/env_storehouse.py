import json
import logging
import operator
import os
import random
from pathlib import Path
from statistics import mean

import gym
import numpy as np
from colorama import Back, Fore, Style

CONF_NAME = "6x6"
MAX_ORDERS = 3
MAX_NUM_BOXES = 6
MIN_NUM_BOXES = 2
FEATURE_NUMBER = 7
MAX_INVALID = 10
MAX_MOVEMENTS = 1000  # 50
MIN_CNN_LEN = 32
AUGMENT_FACTOR = 10
EPISODE = 1


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

    def print_score(self) -> str:
        return (
            f"{self.delivered_boxes}, {self.filled_orders}, {self.clear_run_score}, {self.steps},"
            f"{self.ultra_negative_achieved}, {mean(self.box_ages)},"
            f"{self.non_optimal_material / max(1, self.delivered_boxes) * 100}"
        )


class Box:
    def __init__(self, id: int, position: tuple, type: str = "A", age: int = 0):
        self.id = id
        self.type = type
        self.age = age
        self.position = position

    def update_age(self):
        self.age += 1


class Agent:
    def __init__(self, initial_position: tuple, got_item: int = 0):
        # Only one object at time
        self.position = initial_position
        self.got_item = got_item  # Id of the box it is carrying, 0 if none


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

    def update_entrypoint(self):
        for material in self.material_queue:
            material["timer"] = max(0, material["timer"] - 1)
        self.wait_time_cumulate = max(0, self.wait_time_cumulate - 1)
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
        self.outpoints = outpoints
        self.type_information = type_information
        self.delivery_timer_info = delivery_timer
        self.max_orders = max_orders
        self.max_num_boxes = MAX_NUM_BOXES
        self.min_num_boxes = MIN_NUM_BOXES
        self.delivery_schedule = []  # Form: [{type, timer until ready, num_boxes}]
        self.desired_material = 0
        self.last_delivery_timers = np.Inf

    def reset(self):
        self.delivery_schedule = []
        self.desired_material = 0
        self.last_delivery_timers = np.Inf

    def update_timers(self):
        self.last_delivery_timers += 1
        for delivery in self.delivery_schedule:
            delivery["timer"] = max(0, delivery["timer"] - 1)
        if len(self.delivery_schedule) > 0:
            if self.delivery_schedule[0]["timer"] == 0:
                self.desired_material = self.delivery_schedule[0]["type"]
            else:
                self.desired_material = 0
        else:
            self.desired_material = 0

    def create_order(self, type: str) -> dict:
        timer = round(np.random.poisson(self.type_information[type]["deliver"]["lambda"]))
        num_boxes = random.randrange(self.min_num_boxes, self.max_num_boxes + 1)
        return {"type": type, "timer": timer, "num_boxes": num_boxes}

    def create_delivery(self) -> dict:
        if self.last_delivery_timers <= np.random.poisson(self.delivery_timer_info["lambda"]):
            return None
        if len([order["timer"] for order in self.delivery_schedule if order["timer"] == 0]) > self.max_orders:
            return None
        type = random.choice(list(self.type_information.keys()))
        order = self.create_order(type)
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
    ):
        self.signature = {}
        self.max_id = 1
        self.max_steps = max_steps
        self.max_orders = max_orders
        self.log_flag = logging
        self.load_conf(conf_name)
        if augment is not None:
            self.augmented = augment
        self.random_start = random_start
        self.normalized_state = normalized_state
        self.feature_number = FEATURE_NUMBER + len(self.type_information) - 1
        self.score = Score()
        self.episode = []
        self.available_actions = []
        self.logname = Path(logname)
        self.save_episodes = save_episodes
        self.transpose_state = transpose_state
        if self.augmented:
            size = tuple(dimension * AUGMENT_FACTOR for dimension in self.grid.shape)
        else:
            size = self.grid.shape
        self.action_space = gym.spaces.Discrete(self.grid.shape[0] * self.grid.shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], self.feature_number), dtype=np.uint8)
        self.material = {}  # dict of objects of the class box. {id: Box} form of the dict. ID > 0
        self.restricted_cells = []  # list of coordinates to where the agent cannot move
        self.agents = [Agent((0, 0)) for _ in range(self.num_agents)]
        self.done = False
        self.action = None
        self.num_actions = 0
        self.num_invalid = 0
        self.cum_reward = 0
        self.action_mask = np.zeros(len(list(range(self.action_space.n))))
        if save_episodes:
            self.episode_folder = self.logname.parent / "episodes"
            self.episode_folder.mkdir(parents=True, exist_ok=True)
        if self.log_flag:
            self.logname.mkdir(parents=True, exist_ok=True)
            self.metrics_log = open(str(self.logname / self.logname.name) + "_metrics.csv", "w")
            self.metrics_log.write("Delivered Boxes,Filled orders,Score,Steps,Ultra negative achieved,Mean box ages,Cueles\n")
            # self.actions_log = open(str(self.logname) + "_actions.csv", "w")
            # self.actions_log.write("")

    def __del__(self):
        if self.log_flag:
            self.metrics_log.close()
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

    def outpoints_consume(self):
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

    def calculate_restricted_cells(self):
        """
        Initial estimation of the restricted_cells list, where the agent is forbidden to navigate
        """
        adjacent_cells_delta = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for row in range(1, self.grid.shape[0] - 1):
            for col in range(1, self.grid.shape[0] - 1):
                flag = True
                for delta in adjacent_cells_delta:
                    adjacent_cell = tuple(map(operator.add, (row, col), delta))
                    if self.grid[adjacent_cell] == 0:
                        flag = False
                        break
                if flag:
                    self.restricted_cells.append((row, col))

    def update_restricted_cells(self, ag: Agent):
        """
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
    def get_age_factor(young, old):
        """
        Age substraction bounded within [0, 100]. Returns the percentage of the age factor [0, 1]. Linear (for now)
        """
        return min(max(abs(old - young), 0), 100) / 100

    def get_reward(self, move_status: int, ag: Agent, box: Box = None) -> int:
        if move_status == 0:  # invalid action
            return -1
        elif any(True for element in self.outpoints.delivery_schedule if element["timer"] < 1):
            if ag.position in self.outpoints.outpoints:
                return self.delivery_reward(box)
            try:
                entrypoints_with_items = [
                    ep
                    for ep in [ep for ep in self.entrypoints if len(ep.material_queue) > 0]
                    if ep.material_queue[0]["timer"] == 0
                ]
            except IndexError:
                entrypoints_with_items = []
            if len(self.material) or len(entrypoints_with_items):
                return -0.9
            else:
                return 0
        elif any(
            not (entrypoint.material_queue[0]["timer"])
            for entrypoint in self.entrypoints
            if len(entrypoint.material_queue) > 0
        ):
            return -0.9
        elif len(self.outpoints.delivery_schedule) == 0:
            return 0
        else:
            return 0

    def delivery_reward(self, box):
        min_rew = -0.5
        oldest_box = max(
            (material for material in self.material.values() if material.type == box.type),
            key=operator.attrgetter("age"),
        )

        age_factor = self.get_age_factor(box.age, oldest_box.age)
        if age_factor != 0:
            self.score.non_optimal_material += 1
        self.score.delivered_boxes += 1
        return min_rew * age_factor

    def log(self, action):
        if self.done:
            self.score.box_ages += [box.age for box in self.material.values()]
            if not len(self.score.box_ages):
                self.score.box_ages.append(0)
            self.metrics_log.write(self.score.print_score() + "\n")
            if self.save_episodes:
                with open(f"{self.episode_folder / self.logname.name}_episode_{EPISODE}.json", "w") as f:
                    json.dump(self.episode, f)
        # else:
        # self.actions_log.write(f"{action},{self.agents[0].got_item}\n")

    @staticmethod
    def normalize_age(age: int) -> int:
        return int(min(max(age, 0), 1000) / 1000 * 255)

    def normalize_type(self, type: str) -> int:
        return int((ord(type) - (ord("A") - 1)) * 255 / len(self.type_information))

    def decode_action(self, action: tuple):
        return action[0] * self.grid.shape[0] + action[1]

    def get_available_actions(self) -> None:
        ### Assuming 1 agent
        agent = self.agents[0]
        ####################

        self.action_mask = np.zeros(self.action_mask.shape)

        try:
            ready_to_consume_types = [order["type"] for order in self.outpoints.delivery_schedule if order["timer"] == 0]
        except IndexError:
            ready_to_consume_types = []

        if agent.got_item:  # Agent with item
            free_storage = []
            for ii in range(1, self.grid.shape[0] - 1):
                free_storage.extend(
                    (ii, jj)
                    for jj in range(1, self.grid.shape[1] - 1)
                    if self.grid[ii][jj] == 0 and (ii, jj) != agent.position
                )

            # if not len(free_storage):
            #     raise Exception
            if self.material[agent.got_item].type in ready_to_consume_types:  # Outpoints open
                available_actions = list(self.outpoints.outpoints) + free_storage
            else:  # Outpoints closed
                available_actions = free_storage

        else:  # Agent without item
            try:
                entrypoints_with_items = [
                    ep
                    for ep in [ep for ep in self.entrypoints if len(ep.material_queue) > 0]
                    if ep.material_queue[0]["timer"] == 0
                ]
            except IndexError:
                entrypoints_with_items = []

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

    def get_free_storage_of_grid(self):
        result = []
        for ii in range(1, self.grid.shape[0] - 1):
            result.extend(
                (
                    (ii, jj)
                    for jj in range(1, self.grid.shape[1] - 1)
                    if self.grid[ii][jj] == 0 and (ii, jj) not in self.restricted_cells
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
        self.reset()
        self.done = signature["done"]
        self.agents = signature["agents_raw"]
        self.material = signature["material_raw"]
        self.calculate_restricted_cells()
        self.outpoints.delivery_schedule = signature["outpoints"]["delivery_schedule"]
        self.outpoints.desired_material = signature["outpoints"]["desired_material"]
        self.outpoints.last_delivery_timers = signature["outpoints"]["last_delivery_timers"]
        for ep, info in zip(self.entrypoints, signature["entrypoints"]):
            ep.material_queue = info["material_queue_raw"]
            ep.wait_time_cumulate = info["wait_time_cumulate"]
        self.num_actions = signature["num_actions"]
        for box in list(self.material.values()) + [
            queue[0]["material"]
            for queue in [ep.material_queue for ep in self.entrypoints if len(ep.material_queue) > 0]
            if queue[0]["timer"] == 0
        ]:
            self.grid[box.position] = box.id
        if self.agents[0].got_item:
            self.grid[self.agents[0].position] = 0
        self.max_id = signature["max_id"]
        self.signature = signature

    def get_signature(self) -> dict:
        state = {
            "max_id": self.max_id,
            "done": self.done,
            "boxes": [
                {
                    "id": box.id,
                    "pos": box.position,
                    "age": box.age,
                    "type": box.type,
                }
                for box in self.material.values()
            ],
        }

        state["restricted_cell"] = list(self.restricted_cells)
        state["agents"] = [
            {
                "pos": agent.position,
                "item": self.material[agent.got_item].type if agent.got_item > 0 else 0,
                "item_id": int(agent.got_item) if agent.got_item > 0 else 0,
            }
            for agent in self.agents
        ]
        state["agents_raw"] = self.agents
        state["material_raw"] = self.material
        state["entrypoints"] = [
            {
                "pos": ep.position,
                "queue": [{"timer": item["timer"], "type": item["material"].type} for item in ep.material_queue],
                "material_queue_raw": ep.material_queue,
                "wait_time_cumulate": ep.wait_time_cumulate,
            }
            for ep in self.entrypoints
        ]
        try:
            ready_to_consume_types = [order["type"] for order in self.outpoints.delivery_schedule if order["timer"] == 0]
        except IndexError:
            ready_to_consume_types = []
        state["outpoints"] = {
            "pos": self.outpoints.outpoints,
            "accepted_types": list(set(ready_to_consume_types)),
            "delivery_schedule": self.outpoints.delivery_schedule,
            "desired_material": self.outpoints.desired_material,
            "last_delivery_timers": self.outpoints.last_delivery_timers,
        }
        state["num_actions"] = self.num_actions
        return state

    def save_state_simplified(self, reward: int, action: tuple):
        state = self.get_signature()
        self.episode.append(
            {
                "step": action,
                "reward": reward,
                "cum_reward": self.cum_reward,
                "state": {
                    key: value
                    if key not in ["entrypoints", "outpoints"]
                    else {"pos": value["pos"], "accepted_types": value["accepted_types"]}
                    if key == "outpoints"
                    else {"pos": [pos["pos"] for pos in value]}
                    for key, value in state.items()
                    if key not in ["material_raw", "agents_raw"]
                },
            }
        )

    def get_state(self) -> list:
        box_grid = np.zeros(self.grid.shape)
        restricted_grid = np.zeros(self.grid.shape)
        entrypoint_grid = np.zeros(self.grid.shape)
        outpoint_grids = [np.zeros(self.grid.shape) for _ in self.type_information]
        age_grid = np.zeros(self.grid.shape)
        agent_grid = np.zeros(self.grid.shape)
        agent_material_grid = np.zeros(self.grid.shape)

        for box in self.material.values():
            box_grid[box.position] = self.normalize_type(box.type)

        for cell in self.restricted_cells:
            restricted_grid[cell] = 255

        for entrypoint in self.entrypoints:
            try:
                entrypoint_grid[entrypoint.position] = 255 if entrypoint.material_queue[0]["timer"] == 0 else 0
            except:
                continue

        try:
            ready_to_consume_types = [order["type"] for order in self.outpoints.delivery_schedule if order["timer"] == 0]
        except IndexError:
            ready_to_consume_types = []
        for ii, outpoint_grid in enumerate(outpoint_grids):
            for outpoint in self.outpoints.outpoints:
                outpoint_grid[outpoint] = 255 if sorted(self.type_information.keys())[ii] in ready_to_consume_types else 0

        for box in self.material.values():
            age_grid[box.position] = self.normalize_age(box.age)

        for agent in self.agents:
            agent_grid[agent.position] = 255 if agent.got_item else 128

        for agent in self.agents:
            agent_material_grid[agent.position] = (
                self.normalize_type(self.material[agent.got_item].type) if agent.got_item else 0
            )

        if self.augmented:
            size = tuple(dimension * AUGMENT_FACTOR for dimension in self.grid.shape)
            state_mix = np.array(
                [
                    np.kron(grid, np.ones((AUGMENT_FACTOR, AUGMENT_FACTOR)))
                    for grid in np.array(
                        [
                            box_grid,
                            restricted_grid,
                            entrypoint_grid,
                            age_grid,
                            agent_grid,
                            agent_material_grid,
                        ]
                        + outpoint_grids
                    )
                ]
            )
        else:
            size = self.grid.shape
            state_mix = np.array(
                [box_grid, restricted_grid, entrypoint_grid, age_grid, agent_grid, agent_material_grid] + outpoint_grids
            )
        if self.normalized_state:
            for ii, matrix in enumerate(state_mix):
                state_mix[ii] = matrix / 255
        self.get_available_actions()
        self.signature = self.get_signature()
        if self.transpose_state:
            return state_mix.reshape(size + (self.feature_number,)).transpose([2, 0, 1])
        else:
            return state_mix.reshape(size + (self.feature_number,))

    def move_agent(self, ag: Agent, movement: tuple) -> int:
        """Move an agent to a new position .

        Args:
            ag (Agent): Agent to perform the movement
            movement (tuple): New cell coordinates to interact with

        Returns:
            int: 0 for invalid action.
                 1 for correct take
                 2 for correct drop
                 3 for non optimal action (move to empty cell without object)
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
            assert movement not in self.restricted_cells
            self.num_invalid = 0
        except (AssertionError, IndexError):
            # logging.warning('Invalid movement')
            self.num_invalid += 1
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
        self.update_restricted_cells(ag)  # Update the restricted cell list with the new actions
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
        self.update_restricted_cells(ag)  # Update the restricted cell list with the new actions
        return 1

    # def check_full_occupation(self):
    #     max_occupation = 0.9  # Magic number
    #     return len(self.material) >= (self.grid.shape[0] - 2) * (self.grid.shape[1] - 2) * max_occupation

    def _step(self, action: tuple, render=False) -> list:

        self.num_actions += 1
        info = {"Steps": self.num_actions}
        agent = self.agents[0]  # Assuming 1 agent
        # Done conditions
        if not len(self.available_actions):  # If storehouse full
            self.score.ultra_negative_achieved = True
            self.done = True
        # if self.num_invalid >= MAX_INVALID:
        #     self.score.ultra_negative_achieved = True
        #     self.done = True
        if self.num_actions >= self.max_steps:
            self.done = True
            reward = 0
            info["done"] = "Max movements achieved. Well done!"
            if self.log_flag:
                self.log(action)
            return self.get_state(), reward, self.done, info
        ####

        self.score.steps += 1
        if self.log_flag:
            self.log(action)

        # Update environment with the agent interaction
        if not self.done:
            # Movement
            move_status = self.move_agent(agent, action)
            if move_status in (1, 2):  # If interacted with a Box
                if move_status == 1 and agent.position in [
                    entrypoint.position for entrypoint in self.entrypoints
                ]:  # Added new box into the system
                    box = [entrypoint for entrypoint in self.entrypoints if entrypoint.position == agent.position][
                        0
                    ].get_item()
                    self.material[box.id] = box
                else:
                    box = self.material[self.grid[agent.position] if self.grid[agent.position] > 0 else agent.got_item]
                info["Info"] = f"Box {box.id} moved"
            else:
                box = None
            reward = self.get_reward(move_status, agent, box)
            self.score.clear_run_score += reward
        else:
            info["Info"] = "Done. Please reset the environment"
            reward = -1e3
            return self.get_state(), reward, self.done, info
        # Update environment unrelated to agent interaction
        self.outpoints_consume()
        for box in self.material.values():
            box.update_age()
        order = self.outpoints.create_delivery()
        if order is not None:
            self.max_id = random.choice(self.entrypoints).create_new_order(
                {order["type"]: order["num_boxes"]}, self.max_id
            )  # TODO: Create load balancer
        self.outpoints.update_timers()
        for entrypoint in self.entrypoints:
            self.grid[entrypoint.position] = entrypoint.update_entrypoint()
        info["entrypoint queue"] = [len(entrypoint.material_queue) for entrypoint in self.entrypoints]
        info["outpoint queue"] = self.outpoints.delivery_schedule
        self.cum_reward += reward
        self.save_state_simplified(reward, action)
        if render:
            self.render()
        return self.get_state(), reward, self.done, info

    def step(self, action: int) -> list:
        assert action < self.grid.shape[0] * self.grid.shape[1] and action >= 0
        norm_action = (int(action / self.grid.shape[0]), int(action % self.grid.shape[0]))
        self.action = norm_action
        state, reward, done, info = self._step(norm_action)
        return state, reward, done, info

    def create_random_box(self, position: tuple, type: str = None, age: int = None):
        box = Box(
            id=self.max_id,
            position=position,
            type=type or random.choice(list(self.type_information.keys())),
            age=age or random.choice(range(100)),
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

    def reset(self, render=False) -> list:
        global EPISODE
        EPISODE += 1
        random_flag = self.random_start
        self.signature = {}
        self.restricted_cells = []
        self.episode = []
        self.grid = np.zeros(self.grid.shape)
        self.num_actions = 0

        self.material = {}
        self.outpoints.reset()
        for entrypoint in self.entrypoints:
            entrypoint.reset()

        if random_flag:
            self.reset_random()
        else:
            self.agents = [Agent(initial_position=(3, 3)) for _ in range(self.num_agents)]

        self.calculate_restricted_cells()
        self.done = False
        self.score.reset()
        self.num_invalid = 0
        self.number_actions = 0
        self.cum_reward = 0
        if not len(self.get_available_actions()):
            return self.reset(render)
        if render:
            self.render()
        return self.get_state()

    def reset_random(self):
        box_probability = 0.40  # Magic number
        self.agents = [
            Agent(
                (random.choice(range(1, self.grid.shape[0] - 1)), random.choice(range(1, self.grid.shape[1] - 1))),
                got_item=random.choice([0, 1]),  # If the agent has an item, it will be of ID = 1
            )
            for _ in range(self.num_agents)
        ]
        if self.agents[0].got_item:  # Initialize random box
            self.material[self.agents[0].got_item] = self.create_random_box(position=self.agents[0].position)
        for row in range(1, self.grid.shape[0] - 1):  # Populate the grid of boxes
            for col in range(1, self.grid.shape[1] - 1):
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
        return " " if num == 0 else str(chr(int(ord("A") - 1 + num)))
        # return str(num)

    @staticmethod
    def decode(letter: str) -> int:
        """
        Decodes the grid from letters to numbers
        """
        return ord(letter) - (ord("A") - 1)

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
                    if element > 0 and (r, e) in self.restricted_cells:
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

    env = Storehouse()
    n_a = env.action_space.n
    for _ in range(10):
        env.reset(1)
        # env.render()
        done = False
        t = 0
        while not done and t < 100:
            a = np.random.choice(n_a)
            s, r, done, inf = env.step(a)
            print("State:", s.shape)
            sleep(0.1)
            env.render()
