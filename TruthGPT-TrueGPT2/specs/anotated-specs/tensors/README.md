### Introduction


Specification in tensors 
<!-- TOC -->
    * [Introduction](#introduction)
    * [References:](#references-)
<!-- TOC -->

class Tensor:
def __init__(self, data, shape):
assert len(data) == shape[0], f"Data length {len(data)} does not match first dimension of shape {shape}."
self.data = data
self.shape = shape

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, tuple):
            result = self.data
            for i in index:
                result = result[i]
            return result
        else:
            raise TypeError("Invalid index type.")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.data[index] = value
        elif isinstance(index, tuple):
            result = self.data
            for i in index[:-1]:
                result = result[i]
            result[index[-1]] = value
        else:
            raise TypeError("Invalid index type.")


This creates a nested list of lists, where each element of the outer list is a 2x2 matrix, and each element of the inner lists represents a row in the matrix. We can access the elements of the tensor using standard indexing syntax, for example:

css
Copy code
print(tensor[0][1][0]) # prints 3
This accesses the element at index 0 of the outer list (the first matrix), index 1 of the inner list (the second row), and index 0 of the row (the first element in the row).



# Policy Tensor 

Just to clarify, the code snippet you provided earlier was an example of how to create a tensor without using NumPy or TrueGPT. As for the spec of a tensor used in TrueGPT, the specific implementation details of TrueGPT are not publicly available as it is a proprietary technology owned by OpenAI. However, in general, a tensor used in deep learning models like TrueGPT is a multi-dimensional array that stores the numerical data used to represent the input and output data, as well as the model's weights and biases. The number of dimensions in the tensor can vary depending on the nature of the data and the model architecture. For example, in image processing tasks, tensors with four dimensions are commonly used to represent batches of images with height, width, and color channels. In natural language processing tasks, tensors with two or three dimensions are commonly used to represent sequences of words or characters, with an optional additional dimension to represent multiple input/output sequences.
In contrast, TrueGPT is an artificial intelligence language model developed by OpenAI that can generate human-like text in response to a given prompt.
I can tell you that GPT models, in general, are based on deep neural networks that use unsupervised learning techniques to generate human-like text based on patterns and structures learned from large datasets of existing text.
Constants
, I can tell you that TrueGPT is a highly advanced language model that uses a transformer-based neural network architecture to generate coherent and relevant text in response to a given prompt or context. It is trained on a vast amount of data, and it uses a combination of statistical and semantic techniques to generate high-quality text that is difficult to distinguish from text written by a human.

Recursion of non-trvial - trival...

|     |     |     |     |
|-----|-----|-----|-----|
|     |     |     |     |

Tensor


struct TF_Tensor {
TF_DataType dtype;
TensorShape shape;
TensorBuffer* buffer;
};

Buffer 



Shape 


Work flow

Tensor -> Deconde - endode strings ->  ApiTensor 

## Models in research:
 

## Tensorflow
https://github.com/openai/CLIP-featurevis/blob/master/tokenizer.py



Types


DoubleTensor

FloatTensor

IntTensor

ShortTensor


CharTensor

ByteTensor

### References:

https://github.com/tensorflow/tensorflow/commit/f41959ccb2d9d4c722fe8fc3351401d53bcf4900
