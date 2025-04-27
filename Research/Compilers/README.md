

Triton uses tensor of
pointers as the primary mechanism for memory access. This
means that each element in the tensor is a pointer, representing a block of pointers. Later Triton introduced pointer
to a block tensor (block pointer) to represent a contiguous
block of data. However, in the default compilation pipeline,
all block pointers are eventually rewritten into tensors of
pointers. While this approach is general enough to handle
sparse operations, it necessitates heavy memory analysis to
determine data contiguity. For dense operations, we argue
that using block pointer is a more efficient approach because
it explicitly conveys contiguity information.



## Papers


## Survey 

https://arxiv.org/pdf/2002.03794


## Inference 
https://arxiv.org/pdf/2503.14985


https://arxiv.org/pdf/2405.06907



## Code
https://github.com/agiresearch/CoRE

