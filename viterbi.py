##################################
# #TODO test To do Viterbi decoding
#################################


import numpy as np

def decode(t_matrix, states, start, end, path_len):

    n = len(states)

    # keep the max probs
    probs_matrix = np.zeros((path_len, n))
    # keep the last best state I was at
    prev = np.zeros((path_len-1, n))

    # first state is fixed
    prev[0] = start
    probs_matrix[1] = start
    for p in range(2, path_len):
        for s in range(n):
            probs = probs_matrix[p-1] + t_matrix[:, s]
            prev[p-1][s] = np.argmax(probs) 
            probs_matrix[p][s] = np.max(probs)

    print(probs_matrix)
    print(prev)

    path = np.zeros(path_len)
    path[-1] = end
    path[0] = start
    last_state = end

    for i in range(path_len-2, 1, -1):
        path[i] = prev[i][int(last_state)]
        last_state = prev[i][int(last_state)]
    
    print(path)


t_matrix = np.ones((3,3))
t_matrix = t_matrix / np.sum(t_matrix, axis=1)
states = [0,1,2]
path_len = 5

decode(t_matrix, states, 1, 0, 5)
