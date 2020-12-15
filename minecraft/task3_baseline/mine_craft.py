import numpy as np
import random
from collections import defaultdict as ddict


class MineCraft():
    # actions: right, up, left, down
    direction_deltas = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    num_actions = len(direction_deltas)

    # global variables
    neutral = 1
    wood = 2
    grass = 3
    iron = 4
    gold = 5
    workbench = 6
    toolshed = 7
    obstacles = 8

    def __init__(self, world=2, vanishing=1):
        self._world = world  # select the environment
        self._vanishing = vanishing  # set 1 when objects disappear upon visit and 0 otherwise
        self._x_limit, self._y_limit = self.layout(world).shape

    def layout(self, world):

        if world == 1:
            # world_1 layout
            world_1 = np.ones([10, 10], dtype=int)
            world_1[0][0] = world_1[4][5] = world_1[8][1] = world_1[8][7] = self.grass
            world_1[2][2] = world_1[7][3] = world_1[5][7] = world_1[9][9] = self.wood
            world_1[0][3] = world_1[4][0] = world_1[6][8] = world_1[9][4] = self.iron
            world_1[6][1] = world_1[6][5] = world_1[4][9] = self.workbench
            world_1[2][4] = world_1[9][0] = world_1[7][7] = self.toolshed
            world_1[0][5] = world_1[1][5] = self.obstacles
            world_1[2][5:10] = [self.obstacles] * 5
            world_1[0][7] = self.gold
            return world_1
        elif world == 2:
            # world_2 layout
            world_2 = np.ones([10, 10], dtype=int)
            world_2[0][0] = world_2[4][5] = world_2[8][1] = world_2[8][7] = self.grass
            world_2[2][2] = world_2[7][3] = world_2[5][7] = world_2[9][9] = self.wood
            world_2[0][3] = world_2[4][0] = world_2[6][8] = world_2[9][4] = self.iron
            world_2[6][1] = self.workbench
            world_2[9][0] = self.toolshed
            world_2[0][5] = world_2[1][5] = self.obstacles
            world_2[2][5:10] = [self.obstacles] * 5
            world_2[0][7] = self.gold
            return world_2

    def initialiser(self):
        current_layout = self.layout(self._world)
        while True:
            current_state = np.array([random.randint(0, self._x_limit - 1),
                                      random.randint(0, self._y_limit - 1)])
            if current_layout[current_state[0]][current_state[1]] != self.obstacles:
                break
        return current_state

    def generate_trace(self, episode_number, max_it_number):
        useful_traces = ddict(list)
        useful_paths = ddict(list)
        traces = np.zeros([episode_number, max_it_number, 1], dtype=int)
        paths = np.zeros([episode_number, max_it_number, 3], dtype=int)
        for ep_n in range(episode_number):
            current_state = self.initialiser()
            current_layout = self.layout(self._world)
            for it_n in range(max_it_number):
                traces[ep_n, it_n] = current_layout[current_state[0]][current_state[1]]
                paths[ep_n, it_n, :] = np.array([current_state[0], current_state[1],
                                                 current_layout[current_state[0]][current_state[1]]])

                if (self._vanishing == 1) and \
                        (traces[ep_n, it_n] != self.workbench and traces[ep_n, it_n] != self.toolshed):
                    current_layout[current_state[0]][current_state[1]] = self.neutral
                task_num, trace = self.task_extractor(traces[ep_n, 0:it_n])
                if task_num != 0:
                    if list(trace) not in useful_traces[task_num]:
                        useful_traces[task_num].append(self.remove_duplicates(list(trace)))
                    useful_paths[task_num].append(paths[ep_n, 0:it_n, :].tolist())
                    break
                next_state = self.take_action(current_state, random.randint(0, self.num_actions - 1))
                current_state = next_state
        for i in range(1, 6):
            np.unique(useful_traces[i])
        return useful_traces, useful_paths

    def reward(self, automaton_state):
        if automaton_state == 100:
            return round(10 + random.random()/100, 2)
        else:
            return round(random.random()/100, 2)

    def automaton(self, task, current_automaton_state, label):
        tasks = {1: [[4, 2], [2, 3], [7, 100]],
                 2: [[3, 2], [7, 100]],
                 3: [[2, 2], [3, 3], [4, 4], [7, 100]],
                 4: [[2, 2], [6, 100]],
                 5: [[[3, 2], [4, 3]], [6, 4], [6, 100]],
                 6: [[4, 2], [2, 3], [6, 100]]}
        expected_next_state = tasks[task][current_automaton_state - 1]
        if task == 5 and current_automaton_state == 1:
            if label == 3:
                return 2
            elif label == 4:
                return 3
            else:
                return 1
        else:
            if label == expected_next_state[0]:
                return expected_next_state[1]
            else:
                return current_automaton_state

    def exploration(self, task_number, episode_number, max_it_number):
        task_keys = [[4, 2, 7], [3, 7], [2, 3, 4, 7], [2, 6], [3, 4, 6], [4, 2, 6]]
        # range(1, len(task_keys[task_number - 1]))
        sar_dict = ddict(list)
        for ep_n in range(episode_number):
            current_state = list(self.initialiser())
            current_state.append(1)
            current_layout = self.layout(self._world)
            iter_number = 1
            while current_state[2] != 100 and iter_number < max_it_number:
                iter_number += 1
                if self._vanishing == 1 and \
                        current_layout[current_state[0]][current_state[1]] != self.workbench and \
                        current_layout[current_state[0]][current_state[1]] != self.toolshed:
                    current_layout[current_state[0]][current_state[1]] = self.neutral
                action = random.randint(0, self.num_actions - 1)
                next_state_2d = list(self.take_action(current_state[0:2], action))
                next_automaton_state = self.automaton(task_number,
                                                      current_state[2],
                                                      current_layout[next_state_2d[0]][next_state_2d[1]])
                next_state_2d.append(next_automaton_state)
                sar = [current_state[0], current_state[1], current_state[2],
                       action,
                       next_state_2d[0], next_state_2d[1], next_state_2d[2],
                       self.reward(next_state_2d[2])]
                sar_dict[current_state[2]].append(sar)
                current_state = next_state_2d
        return sar_dict

    def take_action(self, current_state, action_indx):
        next_state = current_state + self.direction_deltas[action_indx]
        # boundaries
        if next_state[0] < 0:
            next_state[0] = 0
        if next_state[0] > self._x_limit - 1:
            next_state[0] = self._x_limit - 1
        if next_state[1] < 0:
            next_state[1] = 0
        if next_state[1] > self._y_limit - 1:
            next_state[1] = self._y_limit - 1
        # obstacles
        if self.layout(self._world)[next_state[0]][next_state[1]] == self.obstacles:
            next_state = current_state
        return next_state

    def task_extractor(self, trace):
        old_trace = trace
        trace = trace[trace > 1]
        # first task [self.wood, self.toolshed]
        # second task [self.grass, self.toolshed]
        # third task [self.wood, self.grass, self.iron, self.toolshed]
        # fourth task [self.wood, self.workbench]
        # fifth task [self.grass, self.workbench]
        # sixth task [self.iron, self.wood, self.workbench]
        if len(trace) > 1:
            if trace[len(trace) - 1] == self.toolshed:
                for i in range(len(trace) - 2, -1, -1):
                    if trace[i] == self.wood:
                        return 1, trace  # trace[i:]
                    elif trace[i] == self.grass:
                        return 2, trace  # trace[i:]
                    elif trace[i] == self.iron:
                        for j in range(i - 1, -1, -1):
                            if trace[j] == self.grass:
                                for k in range(j - 1, -1, -1):
                                    if trace[k] == self.wood:
                                        return 3, trace  # trace[k:]
                                    else:
                                        return 0, old_trace
                            else:
                                return 0, old_trace
                return 0, old_trace
            elif trace[len(trace) - 1] == self.workbench:
                for i in range(len(trace) - 2, -1, -1):
                    if len(trace) > 2 and trace[i] == self.wood:
                        for j in range(i - 1, -1, -1):
                            if trace[j] == self.iron:
                                return 6, trace  # trace[j:]
                            else:
                                return 4, trace  # trace[j:]
                    elif trace[i] == self.wood:
                        return 4, trace  # trace[i:]
                    elif trace[i] == self.grass:
                        return 5, trace  # trace[i:]
                return 0, old_trace
            else:
                return 0, old_trace
        else:
            return 0, old_trace

    def remove_duplicates(self, trace):
        resulting = []
        for i in range(len(trace) - 1, -1, -1):
            if trace[i] not in resulting:
                resulting.append(trace[i])
        return resulting[::-1]
