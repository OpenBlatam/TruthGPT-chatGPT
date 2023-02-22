## Introduction


Anotathed spec for the AI prompt 

### Contents 

Neural Models
Serialization 
GPU -scaling 
Epoch optimized 



Learning Model 

torch modules in dedicated services 

## Description

The torch modules
alpha-0: Working versions of torch, cutorch, nn, cunn, optim fully unit tested with seamless numpy conversions
alpha-1: Serialization to/from disk with sharing intact. initial release of the new neuralnets package based on a Chainer-like design
alpha-2: sharing tensors across processes for hogwild training or data-loading processes. a rewritten optim package for this new nn.
alpha-3: binary installs (prob will take @alexbw 's help here), contbuilds, etc.
alpha-4: a ton of examples across vision, nlp, speech, RL -- this phase might make us rethink parts of the APIs, and hence want to do this in alpha than beta
alpha-5: Putting a simple and efficient story around multi-machine training. Probably simplistic like torch-distlearn. Building the 


### Serialization 



### Math

https://arxiv.org/pdf/1909.00599.pdf

