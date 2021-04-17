from synth.incr_nfa_dfa import make_model
from synth.incr_nfa_dfa import convert_to_dfa_plot
# from synth.incr_nfa_dfa import plot_model
from synth.incr_nfa_dfa import parse_args

import numpy as np
import time
from termcolor import colored


def replace_states_dfa(dfa_model, replace_states):
    for state_pair in replace_states:
        [s1, s2] = state_pair
        for x in dfa_model:
            x[0] = s1 if x[0] == s2 else x[0]
            x[2] = s1 if x[2] == s2 else x[2]

    return dfa_model


def fix_states(old_dfa_model, dfa_model, trace, num_states):
    if not old_dfa_model:
        return dfa_model

    start_state = 1
    for x in dfa_model:
        x[0] = x[0] + num_states if x[0] != start_state else start_state
        x[2] = x[2] + num_states

    state1 = start_state
    state2 = start_state
    replace_states = []
    for x in trace:
        if x == 1:
            state1 = start_state
            state2 = start_state

        if not state1:
            next_state1 = []
        else:
            next_state1 = [y[2] for y in old_dfa_model if y[0] == state1 and y[1] == x]
        next_state2 = [y[2] for y in dfa_model if y[0] == state2 and y[1] == x]

        if next_state1:
            if next_state1[0] == next_state2[0]:
                state1 = next_state1[0]
                state2 = next_state2[0]
                continue
            if next_state1[0] not in [y[0] for y in replace_states]:
                replace_states.append([next_state1[0], next_state2[0]])
            state1 = next_state1[0]
        else:
            state1 = []
        state2 = next_state2[0]

    replace_states = np.unique(replace_states, axis=0)
    replace_states = [list(x) for x in replace_states]
    dfa_model = replace_states_dfa(dfa_model, replace_states)

    new_states = [x[0] for x in dfa_model]
    [new_states.append(x[2]) for x in dfa_model]
    new_states = np.unique(new_states)
    more_than_num_states = [x for x in new_states if x > num_states]
    less_than_num_states = [x for x in new_states if x not in more_than_num_states]

    replace_states = []
    max_state = max(less_than_num_states)
    for x in more_than_num_states:
        replace_states.append([max_state + 1, more_than_num_states[0]])
        if len(more_than_num_states) > 1:
            more_than_num_states = more_than_num_states[1:]
            max_state = max_state + 1
        else:
            break

    dfa_model = replace_states_dfa(dfa_model, replace_states)

    return dfa_model


def process_dfa(dfa_model):
    next_start_state = [x[2] for x in dfa_model if x[0] == 1 and x[1] == 1][0]

    new_model = []
    for x in dfa_model:
        temp = []
        if x[0] == 1:
            temp = [0]
        elif x[0] == next_start_state:
            temp = [1]
        else:
            temp = [x[0]]
        temp.append(x[1])
        if x[2] == 1:
            temp.append(0)
        elif x[2] == next_start_state:
            temp.append(1)
        else:
            temp.append(x[2])
        new_model.append(temp)

    return new_model


def dfa_init():
    hyperparams = parse_args()
    num_states = hyperparams.num_states
    len_seq = hyperparams.window

    var = {'incr': 0, 'events_tup_to_list': [], 'o_event_uniq': [], 'org_trace': [], 'seq_input_uniq': []}
    input_dict = {'event_id': [], 'seq_input_uniq': [], 'event_uniq': [], 'len_seq': len_seq}

    return num_states, var, input_dict, hyperparams


def dfa_update(trace, num_states, dfa_model, nfa_model, model_gen, var, input_dict, hyperparams, start_time, iter_num):
    old_dfa_model = dfa_model.copy()
    old_nfa_model = nfa_model.copy()

    full_events = trace
    start_id = [i for i in range(len(full_events)) if full_events[i] == 'start']
    events_list = []
    for i in range(len(start_id) - 1):
        events_list.append(full_events[start_id[i]:start_id[i + 1] + 1])

    events_tuple = list(set(tuple(x) for x in events_list))
    events_tup_to_list = [list(x) for x in events_tuple]
    var['events_tup_to_list'] = events_tup_to_list

    for event_list in events_tup_to_list:
        [var['org_trace'].append(x) for x in event_list]
        model_gen, var, input_dict, num_states = make_model(event_list, model_gen, var, hyperparams, num_states,
                                                            input_dict, start_time)

    nfa_model = model_gen.copy()
    dfa_model, var, input_dict, num_states = convert_to_dfa_plot(full_events, model_gen, input_dict, num_states,
                                                                 hyperparams, var, start_time, iter_num)
    model_gen = nfa_model.copy()

    dfa_model = fix_states(old_dfa_model, dfa_model, input_dict['event_id'], num_states)
    # plot_model(dfa_model, input_dict, num_states, hyperparams, False, iter_num)

    processed_dfa = process_dfa(dfa_model)
    # plot_model(processed_dfa, input_dict, num_states, hyperparams, False, iter_num)

    return num_states, processed_dfa, dfa_model, nfa_model, model_gen, var, input_dict


def get_next_state(trace, event_uniq, dfa_model):
    temp = []
    try:
        for x in trace:
            temp.append(event_uniq.index(x) + 1)
        start_state = 0
        state = start_state
        for e_id in temp:
            if e_id == 1:
                state = start_state
            next_state = [x[2] for x in dfa_model if x[0] == state and x[1] == e_id]
            if len(next_state) > 1:
                print(colored("[ERROR] Not a DFA", 'magenta'))
                exit(0)
            if not next_state:
                print(colored("[WARNING] No next state found for label " + str(event_uniq[e_id - 1]), 'magenta'))
                return []
            state = next_state[0]

        return state
    except ValueError:
        return -1
