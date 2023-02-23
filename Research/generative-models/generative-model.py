import random

# Define the training data
training_data = ["The cat sat on the mat.",
                 "The dog chased the cat.",
                 "The bird flew over the fence.",
                 "The fish swam in the pond."]

# Define the prompt for zero-shot learning
prompt = "Write a sentence about a mouse."

# Define the vocabulary
vocab = set()
for sentence in training_data:
    vocab.update(sentence.split())
vocab = sorted(vocab)

# Create a dictionary to map words to indices
word_to_index = {word: i for i, word in enumerate(vocab)}

# Convert the training data into sequences
training_sequences = [[word_to_index[word] for word in sentence.split()] for sentence in training_data]

# Pad the sequences to make them of equal length
max_sequence_length = max(len(sequence) for sequence in training_sequences)
padded_sequences = [sequence + [0] * (max_sequence_length - len(sequence)) for sequence in training_sequences]

# Define the input and output sequences for training the model
x_train = padded_sequences[:, :-1]
y_train = padded_sequences[:, 1:]

# Define the RNN model
vocab_size = len(vocab)
embedding_dim = 16
hidden_units = 32
learning_rate = 0.1

Wxh = [[random.gauss(0, 1) / embedding_dim for j in range(hidden_units)] for i in range(vocab_size)]
Whh = [[random.gauss(0, 1) / hidden_units for j in range(hidden_units)] for i in range(hidden_units)]
Why = [[random.gauss(0, 1) / hidden_units for j in range(vocab_size)] for i in range(hidden_units)]

def softmax(x):
    exp_x = [np.exp(x_i - max(x)) for x_i in x]
    return [exp_x_i / sum(exp_x) for exp_x_i in exp_x]

def forward(x):
    h = [0] * hidden_units
    for t in range(len(x)):
        x_t = [0] * vocab_size
        x_t[x[t]] = 1
        a = [sum([x_t[i] * Wxh[i][j] for i in range(vocab_size)]) + sum([h[i] * Whh[i][j] for i in range(hidden_units)]) for j in range(hidden_units)]
        h = [np.tanh(a_i) for a_i in a]
    y = [sum([h[i] * Why[i][j] for i in range(hidden_units)]) for j in range(vocab_size)]
    return softmax(y)

def loss(y, t):
    return -np.log(y[t])

def train(x, t, learning_rate):
    global Wxh, Whh, Why
    h = [0] * hidden_units
    h_list = [h]
    a_list = []
    x_list = []
    for t in range(len(x)):
        x_t = [0] * vocab_size
        x_t[x[t]] = 1
        a = [sum([x_t[i] * Wxh[i][j] for i in range(vocab_size)]) + sum([h[i] * Whh[i][j] for i in range(hidden_units)]) for j in range(hidden_units)]
        h = [np.tanh(a_i) for a_i in a]
        y = [sum([h[i] * Why[i][j] for i in range(hidden_units)]) for j in range(vocab_size)]
        h_list.append(h)
        a_list.append(a)
        x_list.append(x_t)

    # Compute gradients
    dWxh = [[0.0] * hidden_units for i in range(vocab_size)]
    dWhh = [[0.0] * hidden_units for i in range(hidden_units)]
    dWhy = [[0.0] * vocab_size for i in range(hidden_units)]
    dh_next = [0.0] * hidden_units

    for t in reversed(range(len(x))):
        dy = list(h_list[t+1])
        dy[t] -= 1
        for i in range(hidden_units):
            for j in range(vocab_size):
                dWhy[i][j] += h_list[t+1][i] * dy[j]
        dh = [0.0] * hidden_units
        for i in range(hidden_units):
            for j in range(vocab_size):
                dh[i] += dy[j] * Why[i][j]
            for j in range(hidden_units):
                dh[i] += dh_next[j] * Whh[i][j]
        for i in range(hidden_units):
            for j in range(vocab_size):
                dWxh[j][i] += dh[i] * x_list[t][j]
            for j in range(hidden_units):
                dWhh[j][i] += dh[i] * h_list[t][j]
        dh_next = [dh_i * (1 - h_list[t][i]**2) for i, dh_i in enumerate(dh)]

    # Update weights and biases
    for i in range(vocab_size):
        for j in range(hidden_units):
            Wxh[i][j] -= learning_rate * dWxh[i][j]
    for i in range(hidden_units):
        for j in range(hidden_units):
            Whh[i][j] -= learning_rate * dWhh[i][j]
    for i in range(hidden_units):
        for j in range(vocab_size):
            Why[i][j] -= learning_rate * dWhy[i][j]

    # Return updated weights
    return Wxh, Whh, Why
