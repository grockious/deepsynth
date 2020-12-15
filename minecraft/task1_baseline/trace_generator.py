from mine_craft import MineCraft
import pickle as pkl

if __name__ == '__main__':
    mine_craft_env = MineCraft()
    # inputs to generate_trace method: generate_trace(number of episodes, max iteration number)
    traces, states = mine_craft_env.generate_trace(100, 400)
    pkl.dump([traces, states], open("traces_states.p", "wb"))
