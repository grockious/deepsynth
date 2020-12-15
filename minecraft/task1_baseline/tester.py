from mine_craft import MineCraft
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py


if __name__ == '__main__':
    max_it_number = 100
    nn = tf.keras.models.load_model('model_1.h5')
    env = MineCraft(2, 0)
    current_state = list(env.initialiser())
    current_state.append(1)
    current_layout = env.layout(2)
    iter_number = 1
    path = current_state[0:2]
    while current_state[2] != 100 and iter_number < max_it_number:
        iter_number += 1
        neighs = []
        for action in range(4):
            neighs_input = np.append(current_state[0:2], action).reshape(1, 3)
            neighs.append(nn.predict(neighs_input))
        best_action = np.argmax(neighs)
        print(best_action)
        next_state_2d = env.take_action(current_state[0:2], best_action)
        next_automaton_state = env.automaton(1,
                                             current_state[2],
                                             current_layout[next_state_2d[0]][next_state_2d[1]])
        next_state_3d = np.append(next_state_2d, next_automaton_state)
        current_state = next_state_3d
        path = np.vstack((path, current_state[0:2]))
        if current_state[2] == 100:
            print('YAY!')
