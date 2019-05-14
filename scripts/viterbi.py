##################################
# #TODO test To do Viterbi decoding
#################################

import numpy as np

def decode(t_matrix, states, start, end, path_len):

    n = len(states)

    # keep the max probs
    probs_matrix = np.zeros((path_len, n))
    # keep the last best state I was at
    prev = np.zeros((path_len, n))

    # first state is fixed
    prev[0, :] = start
    probs_matrix[0, :] = t_matrix[start, :]

    for p in range(1, path_len):
        for s in range(n):
            probs = probs_matrix[p-1, :] *  t_matrix[:, s] # TODO is t_matrix selected row-wise or column-wise
            #print("Probs {} * {} = {}".format(probs_matrix[p-1, :], t_matrix[:, s], probs))
            prev[p][s] = np.argmax(probs) 
            probs_matrix[p][s] = np.max(probs)
            #print("Probs Matrix {}".format(probs_matrix))
            #print("Prev Matrix {}".format(prev))

    print("**Probs Matrix \n{}".format(probs_matrix))
    print("**Prev Matrix \n{}".format(prev))

    path = np.zeros(path_len)
    path[-1] = end
    path[0] = start

    last_state = end
    for i in range(path_len-2, 0, -1): 
        path[i] = prev[i][int(last_state)]
        last_state = prev[i][int(last_state)]
    
    print(path)
    return path


## t_matrix rows sum to 1
t_matrix = np.array([[0.3, 0.4, 0.3], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
print(t_matrix)
states = [0,1,2]
path_len = 5
start = 1
end = 0

decode(t_matrix, states, start, end, path_len)
