import os
import sys
import tensorflow as tf
from tensorflow import keras
import pickle as pkl
import numpy as np
import random
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import h5py

discount_factor = 0.95
prop = keras.optimizers.RMSprop(lr=0.01)
###
model_1 = keras.Sequential([
    keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid')])
model_1.compile(loss='mean_squared_error',
                metrics=['mean_squared_error'],
                optimizer='Adam')

layout = np.zeros([40, 40], dtype=int)
layout[20:28, 31:39] = np.ones([8, 8], dtype=int) * 1
layout[31:39, 31:39] = np.ones([8, 8], dtype=int) * 2
layout[20:28, 20:28] = np.ones([8, 8], dtype=int) * 3
layout[0:8, 31:39] = np.ones([8, 8], dtype=int) * 4
layout[4:8, 9:13] = np.ones([4, 4], dtype=int) * 5

neutral = 0
objective_1 = 1
objective_2 = 2
objective_3 = 3
objective_4 = 4
unsafe = 5


def automaton(current_automaton_state, label):
    tasks = {1: [[1, 2], [2, 3], [3, 4], [4, 100]],
             2: [[3, 100]]}
    if label == 5:
        return -100
    ###
    expected_next_state = tasks[1][current_automaton_state - 1]
    if label == expected_next_state[0]:
        return expected_next_state[1]
    else:
        return current_automaton_state

    
def reward(automaton_state):
    if automaton_state == 100:
        return round(10 + random.random() / 100, 2)
    else:
        return round(random.random() / 100, 2)


def take_action(current_state, action_indx):
    rand_gen = random.random()
    # actions: right, up, left, down
    direction_deltas = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    if rand_gen > 0.9:  # ignore the action_indx
        next_state = current_state + random.choice(direction_deltas)
    else:
        next_state = current_state + direction_deltas[action_indx]
    # boundaries
    if next_state[0] < 0:
        next_state[0] = 0
    if next_state[0] > 39:
        next_state[0] = 39
    if next_state[1] < 0:
        next_state[1] = 0
    if next_state[1] > 39:
        next_state[1] = 39
    return next_state

# exploration
###
episode_number = 700
###
max_it_number = 500
sar_dict = ddict(list)
trace_list = []
for ep_n in range(episode_number):
    ###
    current_state = [0, 20, 1]
    iter_number = 1
    while current_state[len(current_state) - 1] != 100 and iter_number < max_it_number:
        iter_number += 1
        action = random.randint(0, 3)
        next_state_2d = list(take_action(current_state[0:len(current_state) - 1], action))
        next_state_label = layout[next_state_2d[0]][next_state_2d[1]]
        next_automaton_state = automaton(current_state[len(current_state) - 1], next_state_label)
        if next_automaton_state == -100:
            break
        next_state_2d.append(next_automaton_state)
        sar = current_state + [action] + next_state_2d + [reward(next_state_2d[len(next_state_2d) - 1])]
        sar_dict[current_state[len(current_state) - 1]].append(sar)
        current_state = next_state_2d

sar_1 = np.array(sar_dict[1])
sar_1 = np.unique(sar_1, axis=0)
# models = [model_1, model_2, model_3, model_4]
models = [model_1]
history = ddict(list)
sars = [sar_1]

# refine sars
exp_size = 30
# sar_1 = np.delete(sar_1, random.sample(range(0, len(sar_1)), len(sar_1) - exp_size), axis=0)
# sar_2 = np.delete(sar_2, random.sample(range(0, len(sar_2)), len(sar_2) - exp_size), axis=0)
# sar_3 = np.delete(sar_3, random.sample(range(0, len(sar_3)), len(sar_3) - exp_size), axis=0)
# reward_column = sar_4[:, 7]
# indx_4 = np.where([reward_column > 9])
# high_reward_sar_4 = sar_4[indx_4[1]]
# sar_4 = np.delete(sar_4, random.sample(range(0, len(sar_4)), len(sar_4) - 10 + len(indx_4[1])), axis=0)
# sar_4 = np.vstack((sar_4, high_reward_sar_4))

reward_column = sar_1[:, 7]
indx_1 = np.where([reward_column > 9])
high_reward_sar_1 = sar_1[indx_1[1]]
sar_1 = np.delete(sar_1, random.sample(range(0, len(sar_1)), len(sar_1) - 30 + len(indx_1[1])), axis=0)
sar_1 = np.vstack((sar_1, high_reward_sar_1))

# initialization
###
for i in range(1):
    models[i].fit(np.hstack((sars[i][:, 0:2], sars[i][:, 3:4])), sars[i][:, 7:8], epochs=3, verbose=0)

###
ep = 350
init_state = np.array([0, 20, 1])
utility = []
for i in range(ep):
    print(int(i / ep * 100), '%')
    neighs = []
    for l in range(4):
        neigh_inputs = []
        neigh_inputs = np.append([4, 4], l).reshape(1, 3)
        neighs.append(models[0].predict(neigh_inputs))
    utility.append(max(neighs))
    for j in range(len(models)-1, -1, -1):
        target = np.zeros(len(sars[j]))
        for k in range(len(sars[j])):
            neigh = []
            for l in range(4):
                neigh_input = []
                neigh_input = np.append(sars[j][k, 4:6], l).reshape(1, 3)
                neigh.append(models[min(int(sars[j][k, 6]) - 1, len(models) - 1)].predict(neigh_input))
            target[k] = sars[j][k, 7] + discount_factor * max(neigh)

        history_j = models[j].fit(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4])),
                                  target.reshape(len(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4]))), 1),
                                  epochs=3,
                                  verbose=0)
        history[j].append(history_j)

pkl.dump(sars, open('sars.p', 'wb'))
pkl.dump(utility, open('utility.p', 'wb'))
for i in range(len(models)-1, -1, -1):
    models[i].save('model_' + str(i + 1) + '.h5')
    for j in range(ep):
        pkl.dump(history[i][j].history, open('history/history_' + str(i + 1) + '_' + str(j + 1) + '.p', "wb"))
plt.plot(np.vstack(utility).tolist())
plt.show()