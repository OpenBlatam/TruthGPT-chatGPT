# Multimodal AI

Creation of multimodal components and differents ouputs finalities.


Custom types
We define the following Python custom types for type hinting and readability:

Name	SSZ equivalent	Description
Slot	uint64	a slot number
Audio   uint64
Text    uint64
Video   uint64
Voice
Epoch	uint64	an epoch number
CommitteeIndex	uint64	a committee index at a slot
ValidatorIndex	uint64	a validator registry index
Root	Bytes32	a Merkle root
Hash32	Bytes32	a 256-bit hash
Version	Bytes4	a fork version number
DomainType	Bytes4	a domain type
Domain	Bytes32	a signature domain
BLSPubkey	Bytes48	a BLS12-381 public key
BLSSignature	Bytes96	a BLS12-381 signature

Constants
The following values are (non-configurable) constants used throughout the specification.

Misc
Name	Value
UINT64_MAX	uint64(2**64 - 1)
UINT64_MAX_SQRT	uint64(4294967295)
GENESIS_SLOT	Slot(0)
GENESIS_EPOCH	Epoch(0)
FAR_FUTURE_EPOCH	Epoch(2**64 - 1)
BASE_REWARDS_PER_EPOCH	uint64(4)
DEPOSIT_CONTRACT_TREE_DEPTH	uint64(2**5) (= 32)
JUSTIFICATION_BITS_LENGTH	uint64(4)
ENDIANNESS	'little'
Withdrawal prefixes
Name	Value
BLS_WITHDRAWAL_PREFIX	Bytes1('0x00')
ETH1_ADDRESS_WITHDRAWAL_PREFIX	Bytes1('0x01')
Domain types
Name	Value
DOMAIN_BEACON_PROPOSER	DomainType('0x00000000')
DOMAIN_BEACON_ATTESTER	DomainType('0x01000000')
DOMAIN_RANDAO	DomainType('0x02000000')
DOMAIN_DEPOSIT	DomainType('0x03000000')
DOMAIN_VOLUNTARY_EXIT	DomainType('0x04000000')
DOMAIN_SELECTION_PROOF	DomainType('0x05000000')
DOMAIN_AGGREGATE_AND_PROOF	DomainType('0x06000000')
DOMAIN_APPLICATION_MASK	DomainType('0x00000001')
Note: DOMAIN_APPLICATION_MASK reserves the rest of the bitspace in DomainType for application usage. This means for some DomainType DOMAIN_SOME_APPLICATION, DOMAIN_SOME_APPLICATION & DOMAIN_APPLICATION_MASK MUST be non-zero. This expression for any other DomainType in the consensus specs MUST be zero.


Containers
The following types are SimpleSerialize (SSZ) containers.

Note: The definitions are ordered topologically to facilitate execution of the spec.

Note: Fields missing in container instantiations default to their zero value.

Misc dependencies

Validator
class Validator(Container):
    pubkey: BLSPubkey
    withdrawal_credentials: Bytes32  # Commitment to pubkey for withdrawals
    effective_balance: Gwei  # Balance at stake
    slashed: boolean
    # Status epochs
    activation_eligibility_epoch: Epoch  # When criteria for activation were met
    activation_epoch: Epoch
    exit_epoch: Epoch
    withdrawable_epoch: Epoch  # When validator can withdraw funds

## Papers


## Code

-LLAMA 4

https://github.com/meta-llama/llama-stack/commit/b8f156195650bafef3d9d641a818f16d38cdd45c

