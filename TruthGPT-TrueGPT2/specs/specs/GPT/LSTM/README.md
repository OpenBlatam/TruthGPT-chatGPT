# LSTM
  
Description: Specification for the LSTM (Long Short-Term Memory) model and its components for multimodal AI.

# Phase 0 -- Transformer

<!-- mdformat-toc start --slug=github --no-anchors --maxlevel=6 --minlevel=2 -->

- [Introduction](#introduction)
- [Notation](#notation)
- [Custom types](#custom-types)
- [Constants](#constants)
  - [Misc](#misc)

<!-- mdformat-toc end -->

## Introduction

This document represents the specification for Phase 0 -- The Transformer.

## Notation

Code snippets appearing in `this style` are to be interpreted as Python 3 code.


## Custom types

We define the following Python custom types for type hinting and readability:
| Name | SSZ equivalent | Description |
| ---------------- | -------------- | --------------------------------- |
| Float32 | float32 | 32-bit floating point number |
| Vector | List[Float32]| 1D array of floats |
| Matrix | List[List[Float32]] | 2D array of floats |
| LSTMState | Tuple[Vector, Vector] | Hidden and cell state |


## Constants

The following values are (non-configurable) constants used throughout the specification.

### Misc

Misc
| Name | Value |
| ----------------------------- | --------------------- |
| LSTM_INPUT_SIZE | uint64(256) |
| LSTM_HIDDEN_SIZE | uint64(512) |
| LSTM_NUM_LAYERS | uint64(2) |

## Containers

The following types are [SimpleSerialize (SSZ)](../../ssz/simple-serialize.md) containers.

*Note*: The definitions are ordered topologically to facilitate execution of the spec.

*Note*: Fields missing in container instantiations default to their zero value.

### Misc dependencies

#### `Models`

```python
class Transformer(Container):
    previous_version: Version
    current_version: Version
    model: Model  # model of latest model
```

```python
class LSTM(Container):
    input_size: uint64
    hidden_size: uint64
    num_layers: uint64
    weights_ih: List[Matrix]  # input-hidden weights for each layer
    weights_hh: List[Matrix]  # hidden-hidden weights for each layer
    bias_ih: List[Vector]     # input-hidden bias for each layer
    bias_hh: List[Vector]     # hidden-hidden bias for each layer
```

#### `Validator`

```python
class Validator(Container):
    pubkey: BLSPubkey
    lstm_credentials: Bytes32  # Commitment to pubkey for withdrawals
    runlstm: boolean
    # Status 
    activation_eligibility_lstm: Model # When criteria for activation were met
    activation_lstm: Model
    exit_lstm: Model
```

 ### Model operations


## Helper functions

*Note*: The definitions below are for specification purposes and are not necessarily optimal implementations.

### Math

#### `bytes_to_uint64`

```python
def bytes_to_uint64(data: bytes) -> uint64:
    """
    Return the integer deserialization of ``data`` interpreted as ``ENDIANNESS``-endian.
    """
    return uint64(int.from_bytes(data, ENDIANNESS))
```
 
### Predicates

#### Helper functions

def tanh(x: Float32) -> Float32:
    """
    Compute the hyperbolic tangent activation function.
    """
    return math.tanh(x)

def sigmoid(x: Float32) -> Float32:
    """
    Compute the sigmoid activation function.
    """
    return 1 / (1 + math.exp(-x))

#### Operations

```python
def lstm_step(x: Vector, h_prev: Vector, c_prev: Vector, weights_ih: Matrix, weights_hh: Matrix, bias_ih: Vector, bias_hh: Vector) -> LSTMState:
    """
    Perform a single LSTM step.
    """
    gates = matmul(weights_ih, x) + bias_ih + matmul(weights_hh, h_prev) + bias_hh
    i, f, g, o = split(gates, 4)  # split into input, forget, cell, output gates
    i = sigmoid(i)
    f = sigmoid(f)
    g = tanh(g)
    o = sigmoid(o)
    c = f * c_prev + i * g
    h = o * tanh(c)
    return (h, c)
```