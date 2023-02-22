# Own language

Disclaimer see this as davinci read letters.

Grammar gap 

GADT grammar for a tensor language:
The Tensor class uses a recursive approach to represent an expression tree, where each node is either a scalar value, a variable, or an operation.
```
type ('a, 'b) tensor =
  | Scalar    : float -> ('a, 'a) tensor
  | Var       : 'a -> ('a, 'a) tensor
  | Add       : ('a, 'b) tensor * ('a, 'b) tensor -> ('a, 'b) tensor
  | Mult      : ('a, 'b) tensor * ('b, 'c) tensor -> ('a, 'c) tensor
  | Transpose : ('a, 'b) tensor -> ('b, 'a) tensor
  | Dot       : ('a, 'b) tensor * ('b, 'c) tensor -> ('a, 'c) tensor
  | Inverse   : ('a, 'a) tensor -> ('a, 'a) tensor

```

The variant is AST 



Here is an example GADT grammar of a tensor in VHDL:

```ocaml 
type 'a tensor is
| Scalar : 'a -> 'a tensor
| Vector : 'a list -> 'a tensor
| Matrix : 'a list list -> 'a tensor
| TensorProduct : 'a tensor * 'b tensor -> ('a * 'b) tensor
| TensorSum : 'a tensor * 'a tensor -> 'a tensor
| TensorTranspose : 'a tensor -> 'a tensor
| TensorConjugate : 'a tensor -> 'a tensor


```



## References:

https://dl.acm.org/doi/abs/10.1145/3315454.3329959

