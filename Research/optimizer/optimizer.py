import numpy as np

class Adagrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, w, grad_wrt_w):
        if self.cache is None:
            self.cache = np.zeros_like(w)

        self.cache += np.square(grad_wrt_w)
        w -= (self.learning_rate * grad_wrt_w) / (np.sqrt(self.cache) + self.epsilon)
        return w
