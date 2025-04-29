

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

Overview

Tensor-level IRs have been used by XLA [16] and
Glow [38] to transform tensor programs into predefined LLVM-IR and CUDA-C operation templates (e.g.,
tensor contractions, element-wise operations, etc.) using pattern-matching.
• The polyhedral model [18] has been used by Tensor
Comprehensions (TC) [43] and Diesel [14] to parameterize and automate the compilation of one or many
DNN layers into LLVM-IR and CUDA-C programs.
• Loop synthesizers have been used by Halide [37]
and TVM [10] to transform tensor computations into
loop nests that can be manually optimized using userdefined (though possibly parametric [11]) schedules.

## Papers


## Survey 

https://arxiv.org/pdf/2002.03794

https://arxiv.org/pdf/2311.13587

# Desings 


## Triton Compiler

First Version:
https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

https://dl.acm.org/doi/pdf/10.1145/3623652.3623672

https://dl.acm.org/doi/pdf/10.1145/3623652.3623672

Deep Leraning compilers 
https://www.usenix.org/system/files/osdi18-chen.pdf


## LLVM-based intermediate representation

## Inference 
https://arxiv.org/pdf/2503.14985


https://arxiv.org/pdf/2405.06907



## Code
https://github.com/agiresearch/CoRE

https://github.com/apache/tvm