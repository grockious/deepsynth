from mine_craft import MineCraft
import tensorflow as tf
from tensorflow import keras
import pickle as pkl
import numpy as np
import random
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import h5py

if __name__ == '__main__':
    discount_factor = 0.95
    prop = keras.optimizers.RMSprop(lr=0.01)
    model_1 = keras.Sequential([
        keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation='sigmoid')])
    model_1.compile(loss='mean_squared_error',
                    metrics=['mean_squared_error'],
                    optimizer='Adam')
    model_2 = keras.Sequential([
        keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation='sigmoid')])
    model_2.compile(loss='mean_squared_error',
                    metrics=['mean_squared_error'],
                    optimizer='Adam')
    model_3 = keras.Sequential([
        keras.layers.Dense(128, input_dim=3, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(1, activation='sigmoid')])
    model_3.compile(loss='mean_squared_error',
                    metrics=['mean_squared_error'],
                    optimizer='Adam')

    mine_craft_env = MineCraft()
    # sar means state_action_reward
    sar = mine_craft_env.exploration(1, 500, 1000)
    sar_1 = np.array(sar[1])
    sar_2 = np.array(sar[2])
    sar_3 = np.array(sar[3])
    models = [model_1, model_2, model_3]
    history = ddict(list)
    sars = [sar_1, sar_2, sar_3]

    # refine sars
    exp_size = 1500
    sar_1 = np.delete(sar_1, random.sample(range(0, len(sar_1)), len(sar_1) - exp_size), axis=0)
    sar_2 = np.delete(sar_2, random.sample(range(0, len(sar_2)), len(sar_2) - exp_size), axis=0)
    reward_column = sar_3[:, 7]
    indx_3 = np.where([reward_column > 9])
    high_reward_sar_3 = sar_3[indx_3[1]]
    sar_3 = np.delete(sar_3, random.sample(range(0, len(sar_3)), len(sar_3) - exp_size + len(indx_3[1])), axis=0)
    sar_3 = np.vstack((sar_3, high_reward_sar_3))

    # initialization
    for i in range(3):
        models[i].fit(np.hstack((sars[i][:, 0:2], sars[i][:, 3:4])), sars[i][:, 7:8], epochs=3, verbose=0)

    ep = 100
    init_state = np.array([4, 4, 1])
    utility = []
    for i in range(ep):
        print(int(i / ep * 100), '%')
        neighs = []
        for l in range(4):
            neigh_inputs = []
            neigh_inputs = np.append([4, 4], l).reshape(1, 3)
            neighs.append(models[0].predict(neigh_inputs))
        utility.append(max(neighs))
        for j in range(2, -1, -1):
            target = np.zeros(len(sars[j]))
            for k in range(len(sars[j])):
                neigh = []
                for l in range(4):
                    neigh_input = []
                    neigh_input = np.append(sars[j][k, 4:6], l).reshape(1, 3)
                    neigh.append(models[min(int(sars[j][k, 6]) - 1, 2)].predict(neigh_input))
                target[k] = sars[j][k, 7] + discount_factor * max(neigh)

            history_j = models[j].fit(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4])),
                                      target.reshape(len(np.hstack((sars[j][:, 0:2], sars[j][:, 3:4]))), 1),
                                      epochs=3,
                                      verbose=0)
            history[j].append(history_j)

    pkl.dump(sars, open('sars.p', 'wb'))
    pkl.dump(utility, open('utility.p', 'wb'))
    for i in range(2, -1, -1):
        models[i].save('model_' + str(i + 1) + '.h5')
        for j in range(ep):
            pkl.dump(history[i][j].history, open('history/history_' + str(i + 1) + '_' + str(j + 1) + '.p', "wb"))
plt.plot(np.vstack(utility).tolist())
plt.show()