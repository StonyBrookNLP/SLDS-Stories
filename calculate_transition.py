import sys
import csv
import numpy as np

STATES = ["NEU", "NEG", "POS", "BOS"]


def build_transition(fpath, num_states):

    transitions = []
    with open(fpath) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            #if idx == 3:
                #break
            s = row[-5:]
            s = [STATES.index(x) for x in s]
            s = [3] + s 
            transitions.append(s)

    #print(transitions)
    M = np.zeros((num_states+1, num_states))
    #print(M)

    # collect sum
    for transition in transitions:
        for (i,j) in zip(transition, transition[1:]): # get consecutive states
            M[i][j] += 1

    #print(M)

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [c/s for c in row]

    # rows correspond to states + BOS and columns states
    print(STATES)
    print(M)

    return M


fpath = "rocstories_FULL_sentiment.csv"
num_states = 3

M = build_transition(fpath, num_states)
