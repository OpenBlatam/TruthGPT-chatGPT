# Multimodal AI
  
  Description: Creation of multimodal transformern and components for differents ouputs 
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
| `Slot`           | `uint64`       | a slot number                     |
| `Epoch`          | `uint64`       | an epoch number                   |
| `Text `          | `uint64`       | an  number                        |
| `CommitteeIndex` | `uint64`       | a committee index at a slot       |
| `ValidatorIndex` | `uint64`       | a validator registry index        |
| `Root`           | `Bytes32`      | a Merkle root                     |
| `Hash32`         | `Bytes32`      | a 256-bit hash                    |
| `Version`        | `Bytes4`       | a fork version number             |
| `DomainType`     | `Bytes4`       | a domain type                     |
| `Domain`         | `Bytes32`      | a signature domain                |
| `BLSPubkey`      | `Bytes48`      | a BLS12-381 public key            |
| `BLSSignature`   | `Bytes96`      | a BLS12-381 signature             |

## Constants

The following values are (non-configurable) constants used throughout the specification.

### Misc

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
class Transformer(Container):
    previous_version: Version
    current_version: Version
    model: Model  # model of latest model
```
#### `Validator`

```python
class Validator(Container):
    pubkey: BLSPubkey
    models_credentials: Bytes32  # Commitment to pubkey for withdrawals
    runmodel: boolean
    # Status 
    activation_eligibility_model: Model # When criteria for activation were met
    activation_model: Model
    exit_model: Model
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


#### Operations

## Papers


## Code

-LLAMA 4

https://github.com/meta-llama/llama-stack/commit/b8f156195650bafef3d9d641a818f16d38cdd45c

