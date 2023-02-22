import math

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Define the softmax function
def softmax(x):
    exps = [math.exp(i) for i in x]
    sum_exps = sum(exps)
    return [i / sum_exps for i in exps]

# Define the ReLU activation function
def relu(x):
    return max(0, x)

# Define the Autoregressive Transformer class
class AutoregressiveTransformer:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights and biases
        self.W1 = [[0] * input_size for i in range(hidden_size)]
        self.b1 = [0] * hidden_size
        self.W2 = [[0] * hidden_size for i in range(output_size)]
        self.b2 = [0] * output_size

        # Initialize the gradients
        self.dW1 = [[0] * input_size for i in range(hidden_size)]
        self.db1 = [0] * hidden_size
        self.dW2 = [[0] * hidden_size for i in range(output_size)]
        self.db2 = [0] * output_size

    def forward(self, x):
        # Calculate the first layer activations
        h1 = [0] * len(self.b1)
        for i in range(len(self.W1)):
            for j in range(len(x)):
                h1[i] += self.W1[i][j] * x[j]
            h1[i] += self.b1[i]
            h1[i] = relu(h1[i])

        # Calculate the second layer activations
        h2 = [0] * len(self.b2)
        for i in range(len(self.W2)):
            for j in range(len(h1)):
                h2[i] += self.W2[i][j] * h1[j]
            h2[i] += self.b2[i]

        # Apply the softmax function to the outputs
        y = softmax(h2)

        return y

    def backward(self, x, y_true, y_pred):
        # Calculate the second layer gradients
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                self.dW2[i][j] = (y_pred[i] - y_true[i]) * h1[j]

        # Calculate the first layer gradients
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                sum = 0
                for k in range(len(self.W2)):
                    sum += (y_pred[k] - y_true[k]) * self.W2[k][i]
                self.dW1[i][j] = x[j] * sum * (1 - h1[i]) * h1[i]

        # Update the weights and biases
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                self.W1[i][j] -= self.dW1[i][j] * learning_rate
            self.b1[i] -= self.db1[i] * learning_rate

        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                self.W2[i][j] -= self.dW2[i][j] * learning_rate
            self.b2[i] -= self.db2[i] * learning_rate
