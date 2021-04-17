from minecraft.mine_craft import MineCraft
import tensorflow as tf
from tensorflow import keras
import pickle as pkl
import numpy as np
import random
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import time
import os
import h5py

# SYNTH imports
from synth.synth_wrapper import dfa_init
from synth.synth_wrapper import dfa_update
from synth.synth_wrapper import get_next_state

# Hyper-parameters
task_num = 1
discount_factor = 0.95
exploration_episode_number = 200
max_it_number = 200
episode_number = 100
DFA_UPDATE_FREQ = 5
prop = keras.optimizers.RMSprop(lr=0.01)
###

if __name__ == '__main__':
    MAIN_NFQ = keras.Sequential([
        keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation='sigmoid')])
    MAIN_NFQ.compile(loss='mean_squared_error',
                     metrics=['mean_squared_error'],
                     optimizer='Adam')
    NFQ_dict = {
        1: MAIN_NFQ
    }

    dfa_states = [0, 1]
    set_of_episode_traces = []
    model_gen = []
    nfa_model = []
    dfa_model = []
    num_states, var, input_dict, hyperparams = dfa_init()
    synth_iter_num = 0

    # Exploration
    mine_craft_env = MineCraft()
    sar_dict = ddict(list)
    set_of_episode_traces = []
    for ep_n in range(exploration_episode_number):
        current_state = list(mine_craft_env.initialiser()) + [1]
        current_layout = mine_craft_env.layout(mine_craft_env._world).copy()
        iter_number = 0
        episode_trace = ['start']
        start_time = time.time()
        while mine_craft_env.reward(task_num, np.array(episode_trace[1:])) < 9 \
                and iter_number < max_it_number:
            action = random.randint(0, mine_craft_env.num_actions - 1)
            next_state_2d = list(mine_craft_env.take_action(current_state[0:2], action))
            next_state_label = current_layout[next_state_2d[0]][next_state_2d[1]]
            if next_state_label != mine_craft_env.neutral:  # ignoring the neutral label
                episode_trace.append(next_state_label)
                # ### SYNTH ### #
                old_dfa_states = dfa_states.copy()
                # # SYNTH updates the automaton here:
                if (len(episode_trace) < 3) or \
                        (iter_number % DFA_UPDATE_FREQ == 0) or \
                        (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == -1) or \
                        (get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa) == []):
                    trace = []
                    set_of_episode_traces.append(episode_trace)
                    for x in set_of_episode_traces:
                        trace = trace + x
                    trace = trace + ['start']
                    num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict = dfa_update(
                        trace, num_states,
                        dfa_model,
                        nfa_model,
                        model_gen, var,
                        input_dict,
                        hyperparams,
                        start_time,
                        synth_iter_num)
                    dfa_states = list(set([dfa_transitions[0] for dfa_transitions in processed_dfa] +
                                          [dfa_transitions[2] for dfa_transitions in processed_dfa]))
                    synth_iter_num = synth_iter_num + 1
                    set_of_episode_traces = [episode_trace]

                # Create NFQ modules if necessary
                new_dfa_states = list(set(dfa_states) - set(old_dfa_states))
                if new_dfa_states:
                    for i in new_dfa_states:
                        # Initiate new DQN modules
                        NFQ_dict[i] = keras.Sequential([
                            keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
                            keras.layers.Dense(128, activation=tf.nn.relu),
                            keras.layers.Dense(1, activation='sigmoid')])
                        NFQ_dict[i].compile(loss='mean_squared_error',
                                            metrics=['mean_squared_error'],
                                            optimizer='Adam')
                # Determine next dfa state
                next_automaton_state = get_next_state(episode_trace, input_dict['event_uniq'], processed_dfa)
            else:
                next_automaton_state = current_state[-1]
            if mine_craft_env._vanishing == 1 and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.workbench and \
                    current_layout[current_state[0]][current_state[1]] != mine_craft_env.toolshed:
                current_layout[current_state[0]][current_state[1]] = mine_craft_env.neutral
            next_state_2d.append(next_automaton_state)
            sar = current_state + \
                  [action] + \
                  next_state_2d + \
                  [mine_craft_env.reward(task_num, np.array(episode_trace[1:]))]
            sar_dict[current_state[-1]].append(sar)
            current_state = next_state_2d
            set_of_episode_traces.append(episode_trace)

            iter_number += 1

    sars = []
    models = []
    normal_refinary = []
    rew = None
    for i in range(len(list(sar_dict.keys()))):
        sars.append(np.unique(np.array(sar_dict[list(sar_dict.keys())[i]]), axis=0))
        if sum(sars[i][:, 7] > 9) == 0:
            normal_refinary.append(i)
        else:
            rew = i
        models.append(NFQ_dict[list(sar_dict.keys())[i]])
    history = ddict(list)
    if rew is None:
        print('\n please consider tuning exploration parameters, e.g. increasing exploration_episode_number and '
              'max_it_number \n')

    # refine sars
    exp_size = 1500
    for i in normal_refinary:
        if len(sars[i]) > exp_size:
            sars[i] = np.delete(sars[i], random.sample(range(0, len(sars[i])), len(sars[i]) - exp_size), axis=0)
    reward_column = sars[rew][:, 7]
    indx_rew = np.where([reward_column > 9])
    high_reward_sar_rew = sars[rew][indx_rew[1]]
    sars[rew] = np.delete(sars[rew], random.sample(range(0, len(sars[rew])), len(sars[rew]) - 10 + len(indx_rew[1])),
                          axis=0)
    sars[rew] = np.vstack((sars[rew], high_reward_sar_rew))

    # initialization
    ###
    for i in range(len(models)):
        models[i].fit(np.hstack((sars[i][:, 0:2], sars[i][:, 3:4])), sars[i][:, 7:8], epochs=3, verbose=0)
    ###

    init_state = np.array([4, 4, 1])
    utility = []
    for i in range(episode_number):
        print(int(i / episode_number * 100), '%')
        neighs = []
        for l in range(4):
            neigh_inputs = []
            neigh_inputs = np.append([4, 4], l).reshape(1, 3)
            neighs.append(models[0].predict(neigh_inputs))
        utility.append(max(neighs))
        for j in range(len(models) - 1, -1, -1):
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

    file_path = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(file_path, 'history')
    if not os.path.exists(history_path):
        os.mkdir(history_path)
    pkl.dump(sars, open(os.path.join(history_path, 'sars.p'), 'wb'))
    pkl.dump(utility, open(os.path.join(history_path, 'utility.p'), 'wb'))
    for i in range(len(models) - 1, -1, -1):
        models[i].save(os.path.join(history_path, 'model_' + str(i + 1) + '.h5'))
        for j in range(episode_number):
            pkl.dump(history[i][j].history,
                     open(os.path.join(history_path, 'history_' + str(i + 1) + '_' + str(j + 1) + '.p'), "wb"))
    plt.plot(np.vstack(utility).tolist())
    plt.show()