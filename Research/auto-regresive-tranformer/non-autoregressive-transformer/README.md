# Introduction

This code implements a non-autoregressive transformer model without frameworks. The model consists of an encoder and a decoder. The encoder is used to encode the input sequence into a sequence of hidden states. The decoder is then used to decode the hidden states into a sequence of outputs.

The encoder is a transformer encoder. A transformer encoder is a neural network architecture that is used to encode a sequence of input tokens into a sequence of hidden states. The transformer encoder consists of a stack of self-attention layers. Each self-attention layer takes the hidden states from the previous layer as input and produces a new set of hidden states. The self-attention layers allow the encoder to learn long-range dependencies between the input tokens.

The decoder is a transformer decoder. A transformer decoder is a neural network architecture that is used to decode a sequence of hidden states into a sequence of output tokens. The transformer decoder consists of a stack of self-attention layers and a feed-forward network. The self-attention layers allow the decoder to learn long-range dependencies between the hidden states and the output tokens. The feed-forward network is used to predict the next output token.

The linear layer is used to map the hidden states from the decoder to the output tokens. The output tokens are then passed through a softmax function to produce a probability distribution over the possible output tokens.

The model can be trained using the train method. The train method takes two arguments:

`training_dataset
