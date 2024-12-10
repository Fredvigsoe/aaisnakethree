import numpy as np


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def relu(z):
    return np.maximum(0, z)


def leaky_relu(z):
    epsilon = 0.001
    return np.maximum(z * epsilon, z)
