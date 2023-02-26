import numpy as np

class AutoregressiveTransformer:
    def __init__(self, input_size, hidden_size, output_size, batch_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize the weights and biases
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def forward(self, x):
        # Calculate the first layer activations
        h1 = np.maximum(0, np.dot(self.W1, x) + self.b1)

        # Calculate the second layer activations
        h2 = np.dot(self.W2, h1) + self.b2

        # Apply the softmax function to the outputs
        y = softmax(h2)

        return y

    def backward(self, x, y_true, y_pred):
        # Calculate the second layer gradients
        grad = y_pred - y_true
        self.dW2 = np.dot(grad, self.h1.T)
        self.db2 = np.sum(grad, axis=1, keepdims=True)

        # Calculate the first layer gradients
        grad = np.dot(self.W2.T, grad) * (self.h1 > 0)
        self.dW1 = np.dot(grad, x.T)
        self.db1 = np.sum(grad, axis=1, keepdims=True)

    def update_parameters(self):
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

    def train(self, X, Y, epochs):
        num_batches = X.shape[0] // self.batch_size

        for epoch in range(epochs):
            loss = 0.0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size

                batch_X = X[start_idx:end_idx].T
                batch_Y = Y[start_idx:end_idx].T

                # Forward pass and compute loss
                y_pred = self.forward(batch_X)
                batch_loss = cross_entropy_loss(y_pred, batch_Y)
                loss += batch_loss

                # Backward pass and compute gradients
                self.h1 = np.maximum(0, np.dot(self.W1, batch_X) + self.b1)
                self.backward(batch_X, batch_Y, y_pred)

                # Update parameters
                self.update_parameters()

            # Compute average loss for current epoch
            loss /= X.shape[0]

            print("Epoch %d loss: %.4f" % (epoch, loss))
