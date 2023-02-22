## Generative models

The objective of a generative semi-supervised model is to leverage both labeled and unlabeled data to improve the performance of a classification or regression task.



the generative semi-supervised model should have two main components:

A generative model that can learn the underlying distribution of the data and generate new samples from it. This can be achieved with a variety of models, such as a variational autoencoder, a generative adversarial network, or a mixture density network.

A classifier or regressor that can use both the labeled and unlabeled data to learn the relationship between the input features and the target variable. The classifier or regressor can be a neural network or another type of model that can be trained with both labeled and synthetic data.



## Optimizations

+ Zero shot transfer instead of no fine tunning
