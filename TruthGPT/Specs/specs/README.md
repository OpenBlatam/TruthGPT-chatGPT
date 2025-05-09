# Specifications of Modular AI

This repository hosts the current TruthGPT specifications.
Discussions about design rationale and proposed changes can be brought up and discussed as issues.
Solidified, agreed-upon changes to the specifications can be made through pull requests.

## Specifications

Core specifications for AI clients can be found in [specs](specs). These are
divided into features. Features are researched and developed in parallel, and then consolidated into
sequential upgrades when ready.

### Stable Specifications

| Seq. | Code Name     | Fork Epoch | Links                                                                        |
| ---- | ------------- | ---------- | ---------------------------------------------------------------------------- |
| 0    | **Phase0**    | `0`        | [Specs](specs/phase0), [Tests]
| 1    | **Phase1**    | `1`        | [Specs](specs/phase1), [Tests]d

### In-development Specifications


### Accompanying documents

- [SimpleSerialize (SSZ) spec](ssz/simple-serialize.md)
- [General test format](tests/formats/README.md)

### External specifications

Additional specifications and standards outside of requisite client functionality can be found in
the following repositories:


### Reference tests

Reference tests built from the executable Python spec are available in the [Ethereum Proof-of-Stake
Consensus Spec Tests](https://github.com/ethereum/consensus-spec-tests) repository. Compressed
tarballs are available for each release
[here](https://github.com/ethereum/consensus-spec-tests/releases). Nightly reference tests are
available
[here](https://github.com/ethereum/consensus-specs/actions/workflows/generate_vectors.yml).

## Contributors

### Installation and usage

Clone the repository with:

```bash
git clone 
```

Switch to the directory:

```bash
cd specs
```

View the help output:

```bash
make help
```

### Design goals

The following are the broad design goals for the AI specifications:

The long term is AI for factory mode to adjust for complex generative intruction.

- Minimize complexity, even at the cost of some losses in efficiency.
- Select components that are quantum secure or easily swappable for quantum-secure alternatives.
- Minimize hardware requirements such that a consumer laptop can participate.
- Modular design in low and app devlopment (adapt to langchain)
- Optimize the benchmarks for open and closed LLMs

### Useful resources

- [Design Rationale](https://notes.ethereum.org/s/rkhCgQteN#)
- [Phase0 Onboarding Document](https://notes.ethereum.org/s/Bkn3zpwxB)
- [Online specifications viewer](https://ethereum.github.io/consensus-specs/)
- [PySpec Tests](tests/core/pyspec/README.md)
- [Reference Tests Generators](tests/generators/README.md)
