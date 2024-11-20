import numpy as  np

def mse(groundT, pred):
    # print(groundT,pred)
    return np.mean((groundT-pred) ** 2)

def mse_prime(groundT, pred):
    # print(groundT,pred)
    return -2 * (groundT - pred) / np.size(groundT)

def categorical_cross_entropy(groundT, pred):
    delta = 1e-7  # if pred=0
    pred = np.clip(pred, delta, 1. - delta)
    return -np.sum(groundT * np.log(pred)) / groundT.shape[0]

def categorical_cross_entropy_prime(groundT,pred):
    delta = 1e-7  # if pred=0
    pred = np.clip(pred, delta, 1. - delta)
    return -groundT / pred