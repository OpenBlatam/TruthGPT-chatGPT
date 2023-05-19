import numpy as np

class Model:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.W1 = np.random.randn(hidden_size, input_size)
    self.b1 = np.zeros((hidden_size, 1))
    self.W2 = np.random.randn(output_size, hidden_size)
    self.b2 = np.zeros((output_size, 1))

  def forward(self, x):
    h1 = np.dot(self.W1, x) + self.b1
    h1 = np.maximum(0, h1)
    y = np.dot(self.W2, h1) + self.b2
    return y

  def backward(self, x, y_true):
    y_pred = self.forward(x)
    loss = np.mean(np.square(y_true - y_pred))

    grad_y_pred = 2 * (y_true - y_pred)
    grad_h1 = np.dot(self.W2.T, grad_y_pred)
    grad_h1[h1 < 0] = 0
    grad_W2 = np.dot(grad_y_pred, h1.T)
    grad_b2 = np.sum(grad_y_pred, axis=1, keepdims=True)

    grad_x = np.dot(self.W1.T, grad_h1)
    grad_W1 = np.dot(grad_h1, x.T)
    grad_b1 = np.sum(grad_h1, axis=1, keepdims=True)

    return loss, grad_W1, grad_b1, grad_W2, grad_b2

  def train(self, X, Y, epochs):
    loss_history = []
    for epoch in range(epochs):
      loss, grad_W1, grad_b1, grad_W2, grad_b2 = self.backward(X, Y)
      self.W1 -= self.learning_rate * grad_W1
      self.b1 -= self.learning_rate * grad_b1
      self.W2 -= self.learning_rate * grad_W2
      self.b2 -= self.learning_rate * grad_b2

      loss_history.append(loss)

    return loss_history

  def predict(self, x):
    return self.forward(x)
