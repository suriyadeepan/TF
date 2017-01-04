import numpy as np
from random import sample

'''
 We are gonna generate a dataset - a binary sequence

 1. X : at any t, p(x_t = 1) = 0.5
 2. Y : at t, 
        p(y_t = 1) = 0.5
        p(y_t = 1) = 1    if x_t-3 = 1
        p(y_t = 1) = 0.25 if x_t-8 = 1
        p(y_t = 1) = 0.75 if x_t-8 = 1 and x_t-3 = 1


def gen_data(size=1000000, timesteps=10):
    X = np.random.choice(2, size)
    Y = []
    # generate y randomly, given the conditions
    for i in range(size):
        threshold = 0.5
        if X[i-3]:
            threshold += 0.5
        if X[i-8]:
            threshold -= 0.25
        Y.append(np.random.rand() > threshold)
    # reshape based on number of timesteps
    num_examples = X.shape[0]//timesteps
    return X.reshape([num_examples, timesteps]).astype(np.int32),\
            np.array(Y).reshape([num_examples, timesteps]).astype(np.int32)

'''

def encode(n, seqlen):
    return bin(n)[2:].zfill(seqlen)

def decode(n):
    return int(n,2)

def gen_data(N= 1000000):
    # generate dataset
    seqlen = len(bin(N)[2:])
    X_int = list(range(N))
    X_bin = [ bin(x_int)[2:].zfill(seqlen) for x_int in X_int ]
    Y_int = [ x_bin.count('1') for x_bin in X_bin ]

    # convert strings to indices (0,1)
    X = np.array( [ [int(item) for item in x_bin ] for x_bin in X_bin ] )
    Y = np.array(Y_int)

    return X, Y

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T
