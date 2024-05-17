import math
import numpy as np


def f(i, dot_product):
    return dot_product ** i


def h(x):
    return math.sqrt(x)


def phi(x, w, m):
    res = []
    for i in range(min(len(w), m)):
        for j in range(1, m + 1):
            dot_product = np.dot(w[i], x)
            res.append(f(j, dot_product))
    return h(np.array(res))


x = np.array([1, 2, 3])
w = np.array([[0, 1, 2], [2, 3, 4]])
m = 2

print(phi(x, w, m))