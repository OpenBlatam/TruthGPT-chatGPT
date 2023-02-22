# Compiler of hiw own VDHL tensor



First, the compiler would need to parse the input code and build an abstract syntax tree (AST) that represents the operations to be performed on the tensors. The AST would be built using the constructors defined in the type definition, such as Scalar, Var, Add, etc.

Once the AST has been constructed, the compiler would need to perform type checking to ensure that the types of the tensors involved in each operation are compatible. For example, the types of the operands to Add must be the same, and the number of columns in the first tensor of Mult must match the number of rows in the second tensor.

Next, the compiler would need to generate code to perform each operation on the tensors, using low-level operations such as matrix multiplication and inversion. This code could be generated in C++, using a library such as Eigen to perform the actual tensor operations.

Finally, the compiled code could be executed to evaluate the tensor expressions and produce a result.


What do u need to do basically ?

Implementation of a parser and code generator for a subset of the tensor grammar:
