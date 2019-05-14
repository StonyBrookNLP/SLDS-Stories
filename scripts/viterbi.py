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
    prev[0, :] = start
    probs_matrix[1, :] = t_matrix[:, 0]

    for p in range(2, path_len):
        for s in range(n):
            probs = probs_matrix[p-1, :] + t_matrix[:, s]
            #print("probs {}".format(probs))
            prev[p-1][s] = np.argmax(probs) 
            probs_matrix[p][s] = np.max(probs)
            #print("Probs Matrix {}".format(probs_matrix))
            #print("Prev Matrix {}".format(prev))

    print("**Probs Matrix {}".format(probs_matrix))
    print("**Prev Matrix {}".format(prev))

    path = np.zeros(path_len)
    path[-1] = end
    path[0] = start
    last_state = end

    for i in range(path_len-2, 0, -1):
        print(i)
        path[i] = prev[i][int(last_state)]
        last_state = prev[i][int(last_state)]
    
    print(path)


## running
t_matrix = np.array([[0.3, 0.4, 0.3], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]) #np.ones((3,3))
#t_matrix = t_matrix / np.sum(t_matrix, axis=1)
t_matrix = t_matrix.T
states = [0,1,2]
path_len = 5
start = 1
end = 0

decode(t_matrix, states, start, end, path_len)
