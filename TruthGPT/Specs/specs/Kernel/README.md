# Kernel AI

  Description: Creation of kernel transformer and components for differents ouputs
finalities.

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

| Name             | SSZ equivalent | Description                       |
| ---------------- | -------------- | --------------------------------- |
| `act_quant_kernel`           | `uint64`       | a slot number                     |
| `act_quant`          | `uint64`       | an epoch number                   |
| `weight_dequant_kernel `          | `uint64`       | an  number                        |
| `CommitteeIndex` | `uint64`       | a committee index at a slot       |
| `ValidatorIndex` | `uint64`       | a validator registry index        |
| `Root`           | `Bytes32`      | a Merkle root                     |
| `Hash32`         | `Bytes32`      | a 256-bit hash                    |
| `Version`        | `Bytes4`       | a fork version number             |


## Constants

The following values are (non-configurable) constants used throughout the specification.

### Misc


# Model Configuration

| Name                 | Value    |
|----------------------|----------|
| vocab_size            | 129280   |
| dim                   | 7168     |
| inter_dim             | 18432    |
| moe_inter_dim         | 2048     |
| n_layers              | 61       |
| n_dense_layers        | 3        |
| n_heads               | 128      |
| n_routed_experts      | 256      |
| n_shared_experts      | 1        |
| n_activated_experts   | 8        |
| n_expert_groups       | 8        |
| n_limited_groups      | 4        |
| route_scale           | 2.5      |
| score_func            | sigmoid  |
| q_lora_rank           | 1536     |
| kv_lora_rank          | 512      |
| qk_nope_head_dim      | 128      |
| qk_rope_head_dim      | 64       |
| v_head_dim            | 128      |
| dtype                 | fp8      |


| Name                          | Value                 |
| ----------------------------- | --------------------- |
| `UINT64_MAX`                  | `uint64(2**64 - 1)`   |
| `UINT64_MAX_SQRT`             | `uint64(4294967295)`  |
| `GENESIS_SLOT`                | `Slot(0)`             |
| `GENESIS_EPOCH`               | `Epoch(0)`            |
| `FAR_FUTURE_EPOCH`            | `Epoch(2**64 - 1)`    |
| `BASE_REWARDS_PER_EPOCH`      | `uint64(4)`           |
| `DEPOSIT_CONTRACT_TREE_DEPTH` | `uint64(2**5)` (= 32) |
| `JUSTIFICATION_BITS_LENGTH`   | `uint64(4)`           |
| `ENDIANNESS`                  | `'little'`            |



## Containers

The following types are [SimpleSerialize (SSZ)](../../ssz/simple-serialize.md) containers.

*Note*: The definitions are ordered topologically to facilitate execution of the spec.

*Note*: Fields missing in container instantiations default to their zero value.

### Misc dependencies

#### `Models`

```python
class kernel(Container):
    previous_version: Version
    current_version: Version
    model: Model  # model of latest model
```
#### `Validator`

```python
class Model(BaseModel):
    vocab_size: int
    dim: int
    inter_dim: int
    moe_inter_dim: int
    n_layers: int
    n_dense_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    n_expert_groups: int
    n_limited_groups: int
    route_scale: float
    score_func: str
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    dtype: str
    ```


class Version(BaseModel):
    major: int
    minor: int
    patch: int


 ### Model operations


## Helper functions

*Note*: The definitions below are for specification purposes and are not necessarily optimal implementations.

### Math

#### `Converts FP8 weights to BF16 and saves the converted weights.`

```python
def fp8_to_bf16_path(data: bytes) -> uint64:
    """
    Return the integer deserialization of ``data`` interpreted as ``ENDIANNESS``-endian.
    """
    return uint64(int.from_bytes(data, ENDIANNESS))
```

### Predicates

#### Helper functions


#### Operations

## Papers


## Code

-LLAMA 4

https://github.com/meta-llama/llama-stack/commit/b8f156195650bafef3d9d641a818f16d38cdd45c

