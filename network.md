



Layer network interaction domains
Custom types
We define the following Python custom types for type hinting and readability:

Name	SSZ equivalent	Description
NodeID	uint256	node identifier
SubnetID	uint64	subnet identifier
Constants
Name	Value	Unit
NODE_ID_BITS	256	The bit length of uint256 is 256
Configuration
This section outlines configurations that are used in this spec.

Name	Value	Description
MAX_PAYLOAD_SIZE	10 * 2**20 (= 10485760, 10 MiB)	The maximum allowed size of uncompressed payload in gossipsub messages and RPC chunks
MAX_REQUEST_BLOCKS	2**10 (= 1024)	Maximum number of blocks in a single request
EPOCHS_PER_SUBNET_SUBSCRIPTION	2**8 (= 256)	Number of epochs on a subnet subscription (~27 hours)
MIN_EPOCHS_FOR_BLOCK_REQUESTS	MIN_VALIDATOR_WITHDRAWABILITY_DELAY + CHURN_LIMIT_QUOTIENT // 2 (= 33024, ~5 months)	The minimum epoch range over which a node must serve blocks
ATTESTATION_PROPAGATION_SLOT_RANGE	32	The maximum number of slots during which an attestation can be propagated
MAXIMUM_GOSSIP_CLOCK_DISPARITY	500	The maximum milliseconds of clock disparity assumed between honest nodes
MESSAGE_DOMAIN_INVALID_SNAPPY	DomainType('0x00000000')	4-byte domain for gossip message-id isolation of invalid snappy messages
MESSAGE_DOMAIN_VALID_SNAPPY	DomainType('0x01000000')	4-byte domain for gossip message-id isolation of valid snappy messages
SUBNETS_PER_NODE	2	The number of long-lived subnets a beacon node should be subscribed to
ATTESTATION_SUBNET_COUNT	2**6 (= 64)	The number of attestation subnets used in the gossipsub protocol.
ATTESTATION_SUBNET_EXTRA_BITS	0	The number of extra bits of a NodeId to use when mapping to a subscribed subnet
ATTESTATION_SUBNET_PREFIX_BITS	int(ceillog2(ATTESTATION_SUBNET_COUNT) + ATTESTATION_SUBNET_EXTRA_BITS)
MAX_CONCURRENT_REQUESTS	2	Maximum number of concurrent requests per protocol ID that a client may issue
MetaData
Clients MUST locally store the following MetaData:

(
  seq_number: uint64
  attnets: Bitvector[ATTESTATION_SUBNET_COUNT]
)
Where

seq_number is a uint64 starting at 0 used to version the node's metadata. If any other field in the local MetaData changes, the node MUST increment seq_number by 1.
ttnets is a Bitvector representing the node's persistent attestation subnet subscriptions.
Note: MetaData.seq_number is used for versioning of the node's metadata, is entirely independent of the ENR sequence number, and will in most cases be out of sync with the ENR sequence number.


