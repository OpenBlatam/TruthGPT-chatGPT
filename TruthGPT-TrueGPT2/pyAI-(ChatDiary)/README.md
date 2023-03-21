
## Reality

This is how the code is really looks like:
https://labs.openai.com/e/y8losgb5ikuzPT9XMqJOt9Me/Xq19NMhbE3jJYHjMumScOiLj


The code defines a class called Model, which is the main component of the GPT-2 model. The Model class contains several layers that are used to process and transform the input text data. The Block class represents the core of the GPT-2 model, and it contains the attention mechanism, which allows the model to attend to relevant parts of the input sequence. The Attention class is used to compute the attention scores between the query and key vectors.

The Norm class is used to normalize the input data and apply a diagonal affine transform. The MLP class is a multi-layer perceptron that is used to transform the input data into a higher dimensional space. The Conv1D class represents a one-dimensional convolution operation.

The position_for function is used to compute the position embeddings that are used by the model to capture the position of each token in the input sequence. The gelu function implements the Gaussian Error Linear Units activation function, which is used in the GPT-2 model.

The code also defines an HParams class that specifies the hyperparameters of the model, such as the number of hidden units, the number of layers, and the size of the vocabulary.

## Spec



### Redutcion
BPE algorithm in the object
https://github.com/lizihan97/BPE

Gauss
x
### Reasearch in model of LMs
https://github.com/legacyai/tf-transformers/tree/main/research

Papers of variants

https://arxiv.org/pdf/1810.04805.pdf
