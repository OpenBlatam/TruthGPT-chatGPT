import numpy as np


def attention(Q, K, V, dk):
    QKT = np.dot(Q, K.T)
    weighted_values = np.dot(np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 0, QKT / np.sqrt(dk)), V)
    return weighted_values


Q = np.random.rand(2, 2)
K = np.random.rand(2, 2)
V = np.random.rand(2, 2)
dk = np.shape(Q)[1]

outputs = attention(Q, K, V, dk)
print(outputs)