# Autoregressive Transformer GPT Model

## Description

A transformer is a type of neural network architecture that is often used for sequence-to-sequence tasks, such as machine translation or text summarization. It was introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017) and has become popular in natural language processing tasks. A transformer consists of an encoder and a decoder, each of which contains multiple layers of self-attention and feedforward neural networks. The self-attention mechanism allows the model to attend to different parts of the input sequence, and the feedforward neural networks provide non-linear transformations. The transformer has achieved state-of-the-art results on a variety of natural language processing tasks

This repository contains a Python implementation of a two-layer neural network with ReLU and softmax activation functions, and an autoregressive transformer model.

## Usage

To use the code, import the necessary functions and classes from the module:

```python
from autoregressive_transformer import AutoregressiveTransformer, relu, softmax
Then, create an instance of the AutoregressiveTransformer class with appropriate input, hidden, and output sizes:

python
Copy code
model = AutoregressiveTransformer(input_size=2, hidden_size=3, output_size=2)
Perform a forward pass on some input data x to obtain a predicted output vector y_pred:

python
Copy code
y_pred = model.forward(x)
Perform a backward pass on the input data x, true output data y_true, and predicted output data y_pred to update the weights and biases of the neural network using backpropagation:

python
Copy code
model.backward(x, y_true, y_pred, learning_rate=0.1)
The learning_rate parameter specifies the rate at which the weights and biases are updated.

API
The module provides the following functions and classes:

sigmoid(x)
This function takes a scalar input x and returns the sigmoid function of x, which is defined as 1 / (1 + exp(-x)).

softmax(x)
This function takes a list or array of numerical values x and returns the softmax function of x, which is defined as exp(x_i) / sum(exp(x)) for each element x_i of x. The output is a list or array of the same length as x.

relu(x)
This function takes a scalar input x and returns the rectified linear unit (ReLU) function of x, which is defined as max(0, x).

class AutoregressiveTransformer
This class implements a two-layer neural network with ReLU activation in the first layer and softmax activation in the second layer. The constructor takes three arguments: input_size, hidden_size, and output_size, which specify the sizes of the input, hidden, and output layers, respectively.

The class has the following methods:

__init__(self, input_size, hidden_size, output_size)
This method initializes the weights and biases of the network to zero, and initializes the gradients to zero as well.

forward(self, x)
This method takes an input vector x and performs a forward pass through the neural network. It calculates the first layer activations using the ReLU activation function, and the second layer activations using the softmax activation function. It returns the output vector y.

backward(self, x, y_true, y_pred, learning_rate)
This method takes an input vector x, a true output vector y_true, a predicted output vector y_pred, and a learning rate, and performs a backward pass through the neural network to update the weights and biases using backpropagation. It calculates the gradients of the loss with respect to the weights and biases of each layer, and updates the weights and biases by subtracting the gradient multiplied by the learning rate.

License
This code is licensed under the MIT License.

css
Copy code

You can then commit the README file to your repository and push the changes to GitHub. The README file will be
