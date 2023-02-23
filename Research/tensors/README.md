
Constants

float64 = 100 M (need a 100x)



There is an example of the specification of a 3-dimensional tensor:

Shape: (3, 4, 2)
Data Type: Float32
Number of Dimensions: 3
Number of Elements: 24 (3 x 4 x 2)
Memory Size: 96 bytes (4 bytes per element x 24 elements)
Element Access: tensor[i][j][k] where i ranges from 0 to 2, j ranges from 0 to 3, and k ranges from 0 to 1.
In this example, the tensor has three dimensions with lengths 3, 4, and 2 respectively. The data type is float32, meaning each element of the tensor is a 32-bit floating-point number. The total number of elements is 24, and the memory size is 96 bytes (assuming 4 bytes per element).

To access an element of the tensor, you would need to provide three indices representing the position of the element in each dimension. For example, to access the element at position (1, 2, 0), you would use the expression tensor[1][2][0].
