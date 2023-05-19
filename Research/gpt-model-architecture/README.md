# GPT Model
![](https://miro.medium.com/v2/resize:fit:1356/0*sAWvrBRO6CyqrwKL)
## Description

The Generative Trained Model is a model:
state-of-the-art GPT models, such as layer normalization, positional encoding, or attention masking.

```
Compile model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
```
## 


## Design algorithm

Increase the number of layers and/or the hidden size: The original GPT model had 12 layers and 768 hidden units, but subsequent versions have used up to 48 layers and 1536 hidden units. More layers and hidden units can potentially improve the model's ability to learn complex patterns in the input.

Use a larger training corpus: The original GPT model was trained on the WebText corpus, which consisted of about 40 GB of text. Subsequent versions have used larger corpora, such as the BookCorpus and Common Crawl datasets. Using more training data can help the model learn more nuanced patterns in the text.

Use a more powerful language model architecture: Since the release of the original GPT model, several other architectures have been developed that improve on its performance, such as GPT-2, GPT-3, and RoBERTa. These models incorporate various improvements such as larger training corpora, better attention mechanisms, and more effective pre-training methods.

Incorporate more context into the input: While the original GPT model was trained to predict the next word in a sequence based on a fixed window of input tokens, subsequent models have used more sophisticated input representations that take into account larger contexts, such as entire documents or even multiple documents.

Fine-tune the model on specific tasks: While the pre-trained GPT models are highly effective at generating text, they can be further improved by fine-tuning them on specific tasks, such as text classification or question answering. This involves re-training the model on a smaller dataset that is specific to the target task.

## Scaelability



## Botlenecks 


## State of the art 

### Math


## Update weights and biases

## Compute gradients

## Define the RNN model

Parameters:

# Define the RNN model
vocab_size = len(vocab)
embedding_dim = 16
hidden_units = 32
learning_rate = 0.1


## Specs 

### Architecture 



## References:

### Code:

https://github.com/cyk1337/Transformer-in-PyTorch/blob/master/transformer_xl/Transformer_xl.py

### Literature:

https://towardsdatascience.com/examining-the-transformer-architecture-part-1-the-openai-gpt-2-controversy-feceda4363bb


