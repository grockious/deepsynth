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
        keras.layers.Dense(128 * 3, input_dim=3, activation=tf.nn.relu),
        keras.layers.Dense(128 * 3, activation=tf.nn.relu),
        keras.layers.Dense(128 * 3, activation=tf.nn.relu),
        keras.layers.Dense(128 * 3, activation=tf.nn.relu),
        keras.layers.Dense(1, activation='sigmoid')])
    model_1.compile(loss='mean_squared_error',
                    metrics=['mean_squared_error'],
                    optimizer='Adam')

    mine_craft_env = MineCraft()
    # sar means state_action_reward
    sar_o = mine_craft_env.exploration(3, 500, 1000)
    sar_1 = np.array(sar_o[1])
    sar_2 = np.array(sar_o[2])
    sar_3 = np.array(sar_o[3])
    sar_4 = np.array(sar_o[4])
    history = ddict(list)
    sar = []

    # refine sars
    exp_size = 20000
    sar_1 = np.delete(sar_1, random.sample(range(0, len(sar_1)), len(sar_1) - exp_size), axis=0)
    sar_2 = np.delete(sar_2, random.sample(range(0, len(sar_2)), len(sar_2) - exp_size), axis=0)
    sar_3 = np.delete(sar_3, random.sample(range(0, len(sar_3)), len(sar_3) - exp_size), axis=0)
    sar_4 = np.delete(sar_4, random.sample(range(0, len(sar_4)), len(sar_4) - exp_size), axis=0)

    sar = np.vstack((sar_1, sar_2))
    sar = np.vstack((sar, sar_3))
    sar = np.vstack((sar, sar_4))
    # initialization
    model_1.fit(np.hstack((sar[:, 0:2], sar[:, 3:4])), sar[:, 7:8], epochs=3, verbose=0)

    ep = 100
    init_state = np.array([4, 4, 1])
    utility = []
    for i in range(ep):
        print(int(i / ep * 100), '%')
        neighs = []
        for l in range(4):
            neigh_inputs = []
            neigh_inputs = np.append([4, 4], l).reshape(1, 3)
            neighs.append(model_1.predict(neigh_inputs))
        utility.append(max(neighs))
        for j in range(1):
            target = np.zeros(len(sar))
            for k in range(len(sar)):
                neigh = []
                for l in range(4):
                    neigh_input = []
                    neigh_input = np.append(sar[k, 4:6], l).reshape(1, 3)
                    neigh.append(model_1.predict(neigh_input))
                target[k] = sar[k, 7] + discount_factor * max(neigh)

            history_j = model_1.fit(np.hstack((sar[:, 0:2], sar[:, 3:4])),
                                    target.reshape(len(np.hstack((sar[:, 0:2], sar[:, 3:4]))), 1),
                                    epochs=3,
                                    verbose=0)
            history[j].append(history_j)

    pkl.dump(sar, open('sars.p', 'wb'))
    pkl.dump(utility, open('utility.p', 'wb'))
    for i in range(1):
        model_1.save('model_' + str(i + 1) + '.h5')
        for j in range(ep):
            pkl.dump(history[i][j].history, open('history/history_' + str(i + 1) + '_' + str(j + 1) + '.p', "wb"))
plt.plot(np.vstack(utility).tolist())
plt.show()