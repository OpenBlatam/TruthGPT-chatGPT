"""
Blockchain-Based Router
=======================

Advanced routing using blockchain technology for expert verification,
consensus mechanisms, and decentralized decision making.
"""

import base64
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from optimization_core.modules.feed_forward.routing.modular_routing.base_router import (
    BaseRouter,
    RouterConfig,
    RoutingResult,
)

# Configure logging
logger = logging.getLogger(__name__)


class Blockchain:
    """Blockchain implementation for expert verification."""

    def __init__(self, difficulty: int = 4) -> None:
        """
        Initialize Blockchain.

        Args:
            difficulty: Mining difficulty.
        """
        self.difficulty = difficulty
        self.chain: List[Dict[str, Any]] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.nodes: set = set()
        self.consensus_threshold = 0.51

        # Create genesis block
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Create the genesis block."""
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': [],
            'previous_hash': '0',
            'nonce': 0,
            'hash': self._calculate_hash(0, time.time(), [], '0', 0)
        }
        self.chain.append(genesis_block)

    def _calculate_hash(
        self,
        index: int,
        timestamp: float,
        transactions: List[Dict[str, Any]],
        previous_hash: str,
        nonce: int
    ) -> str:
        """
        Calculate hash for a block.

        Args:
            index: Block index.
            timestamp: Block timestamp.
            transactions: List of transactions.
            previous_hash: Hash of previous block.
            nonce: Mining nonce.

        Returns:
            Calculated hash string.
        """
        block_string = f"{index}{timestamp}{json.dumps(transactions, sort_keys=True)}{previous_hash}{nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def _proof_of_work(
        self,
        index: int,
        timestamp: float,
        transactions: List[Dict[str, Any]],
        previous_hash: str
    ) -> Tuple[int, str]:
        """
        Mine a new block using proof of work.

        Args:
            index: Block index.
            timestamp: Block timestamp.
            transactions: List of transactions.
            previous_hash: Hash of previous block.

        Returns:
            Tuple of (nonce, hash).
        """
        nonce = 0
        while True:
            hash_value = self._calculate_hash(index, timestamp, transactions, previous_hash, nonce)
            if hash_value.startswith('0' * self.difficulty):
                return nonce, hash_value
            nonce += 1

    def add_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Add a transaction to the pending transactions.

        Args:
            transaction: Transaction data.

        Returns:
            True if valid and added.
        """
        if self._validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            return True
        return False

    def _validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate a transaction."""
        required_fields = ['from', 'to', 'amount', 'timestamp', 'signature']
        return all(field in transaction for field in required_fields)

    def mine_block(self, miner_address: str) -> Optional[Dict[str, Any]]:
        """
        Mine a new block.

        Args:
            miner_address: Address of the miner.

        Returns:
            Mined block or None if no pending transactions.
        """
        if not self.pending_transactions:
            return None

        previous_block = self.chain[-1]
        index = len(self.chain)
        timestamp = time.time()

        # Mine block
        nonce, hash_value = self._proof_of_work(
            index, timestamp, self.pending_transactions, previous_block['hash']
        )

        # Create block
        block = {
            'index': index,
            'timestamp': timestamp,
            'transactions': self.pending_transactions.copy(),
            'previous_hash': previous_block['hash'],
            'nonce': nonce,
            'hash': hash_value,
            'miner': miner_address
        }

        # Add to chain
        self.chain.append(block)

        # Clear pending transactions
        self.pending_transactions = []

        return block

    def validate_chain(self) -> bool:
        """
        Validate the entire blockchain.

        Returns:
            True if valid.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check hash
            if current_block['hash'] != self._calculate_hash(
                current_block['index'],
                current_block['timestamp'],
                current_block['transactions'],
                current_block['previous_hash'],
                current_block['nonce']
            ):
                return False

            # Check previous hash
            if current_block['previous_hash'] != previous_block['hash']:
                return False

        return True

    def get_balance(self, address: str) -> float:
        """
        Get balance for an address.

        Args:
            address: Wallet address.

        Returns:
            Balance.
        """
        balance = 0.0
        for block in self.chain:
            for transaction in block['transactions']:
                if transaction['from'] == address:
                    balance -= transaction['amount']
                if transaction['to'] == address:
                    balance += transaction['amount']
        return balance

    def get_transaction_history(self, address: str) -> List[Dict[str, Any]]:
        """
        Get transaction history for an address.

        Args:
            address: Wallet address.

        Returns:
            List of transactions.
        """
        transactions = []
        for block in self.chain:
            for transaction in block['transactions']:
                if transaction['from'] == address or transaction['to'] == address:
                    transactions.append(transaction)
        return transactions


class SmartContract:
    """Smart contract for expert verification and routing."""

    def __init__(self, contract_address: str) -> None:
        """
        Initialize SmartContract.

        Args:
            contract_address: Address of the contract.
        """
        self.contract_address = contract_address
        self.experts: Dict[str, Dict[str, Any]] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.consensus_threshold = 0.51
        self.verification_required = True

    def register_expert(
        self,
        expert_id: str,
        expert_data: Dict[str, Any],
        signature: str
    ) -> bool:
        """
        Register a new expert.

        Args:
            expert_id: ID of the expert.
            expert_data: Expert data.
            signature: Digital signature.

        Returns:
            True if registered.
        """
        if self._verify_signature(expert_data, signature):
            self.experts[expert_id] = {
                'data': expert_data,
                'signature': signature,
                'registration_time': time.time(),
                'verification_status': 'pending',
                'reputation_score': 0.0,
                'total_routes': 0,
                'successful_routes': 0
            }
            return True
        return False

    def verify_expert(self, expert_id: str, verification_data: Dict[str, Any]) -> bool:
        """
        Verify an expert.

        Args:
            expert_id: Expert ID.
            verification_data: Data for verification.

        Returns:
            True if verified.
        """
        if expert_id not in self.experts:
            return False

        # Perform verification checks
        verification_passed = self._perform_verification_checks(expert_id, verification_data)

        if verification_passed:
            self.experts[expert_id]['verification_status'] = 'verified'
            self.experts[expert_id]['reputation_score'] = 1.0
            return True
        else:
            self.experts[expert_id]['verification_status'] = 'failed'
            return False

    def _verify_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """Verify digital signature."""
        # Simplified signature verification
        # In practice, this would use proper cryptographic verification
        return len(signature) > 10  # Basic validation

    def _perform_verification_checks(
        self,
        expert_id: str,
        verification_data: Dict[str, Any]
    ) -> bool:
        """Perform verification checks for an expert."""
        # Simplified verification checks
        required_fields = ['capabilities', 'performance_metrics', 'certifications']
        return all(field in verification_data for field in required_fields)

    def route_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a routing request.

        Args:
            request_data: Request data.

        Returns:
            Routing transaction data.
        """
        # Find suitable experts
        suitable_experts = self._find_suitable_experts(request_data)

        if not suitable_experts:
            return {'error': 'No suitable experts found'}

        # Select best expert based on reputation and capabilities
        best_expert = self._select_best_expert(suitable_experts, request_data)

        if best_expert is None:
             return {'error': 'No suitable experts found'}

        # Create routing transaction
        routing_transaction = {
            'request_id': request_data.get('request_id', ''),
            'expert_id': best_expert,
            'timestamp': time.time(),
            'routing_decision': 'accepted',
            'confidence': self.experts[best_expert]['reputation_score']
        }

        # Record routing
        self.routing_history.append(routing_transaction)
        self.experts[best_expert]['total_routes'] += 1

        return routing_transaction

    def _find_suitable_experts(self, request_data: Dict[str, Any]) -> List[str]:
        """Find experts suitable for the request."""
        suitable_experts = []

        for expert_id, expert_info in self.experts.items():
            if expert_info['verification_status'] == 'verified':
                # Check if expert has required capabilities
                if self._expert_has_capabilities(expert_info, request_data):
                    suitable_experts.append(expert_id)

        return suitable_experts

    def _expert_has_capabilities(
        self,
        expert_info: Dict[str, Any],
        request_data: Dict[str, Any]
    ) -> bool:
        """Check if expert has required capabilities."""
        required_capabilities = request_data.get('required_capabilities', [])
        expert_capabilities = expert_info['data'].get('capabilities', [])

        return all(cap in expert_capabilities for cap in required_capabilities)

    def _select_best_expert(
        self,
        suitable_experts: List[str],
        request_data: Dict[str, Any]
    ) -> Optional[str]:
        """Select the best expert from suitable experts."""
        if not suitable_experts:
            return None

        # Score experts based on reputation and performance
        expert_scores = {}
        for expert_id in suitable_experts:
            expert_info = self.experts[expert_id]
            score = expert_info['reputation_score']

            # Bonus for recent successful routes
            total = expert_info['total_routes']
            if total > 0:
                success_rate = expert_info['successful_routes'] / total
                score += success_rate * 0.5

            expert_scores[expert_id] = score

        # Return expert with highest score
        return max(expert_scores, key=lambda k: expert_scores[k])

    def update_expert_reputation(self, expert_id: str, success: bool) -> None:
        """Update expert reputation based on performance."""
        if expert_id in self.experts:
            if success:
                self.experts[expert_id]['successful_routes'] += 1
                # Increase reputation
                self.experts[expert_id]['reputation_score'] = min(1.0,
                    self.experts[expert_id]['reputation_score'] + 0.1)
            else:
                # Decrease reputation
                self.experts[expert_id]['reputation_score'] = max(0.0,
                    self.experts[expert_id]['reputation_score'] - 0.05)

    def get_expert_stats(self, expert_id: str) -> Dict[str, Any]:
        """Get statistics for an expert."""
        if expert_id not in self.experts:
            return {}

        expert_info = self.experts[expert_id]
        total = expert_info['total_routes']
        return {
            'expert_id': expert_id,
            'verification_status': expert_info['verification_status'],
            'reputation_score': expert_info['reputation_score'],
            'total_routes': total,
            'successful_routes': expert_info['successful_routes'],
            'success_rate': expert_info['successful_routes'] / max(total, 1),
            'registration_time': expert_info['registration_time']
        }


class ConsensusMechanism:
    """Consensus mechanism for decentralized decision making."""

    def __init__(self, consensus_type: str = "proof_of_stake") -> None:
        """
        Initialize ConsensusMechanism.

        Args:
            consensus_type: Type of consensus.
        """
        self.consensus_type = consensus_type
        self.validators: Dict[str, Dict[str, Any]] = {}
        self.stake_pool: Dict[str, float] = {}
        self.consensus_threshold = 0.51

    def add_validator(self, validator_id: str, stake: float) -> bool:
        """Add a validator to the consensus mechanism."""
        self.validators[validator_id] = {
            'stake': stake,
            'voting_power': stake,
            'reputation': 1.0,
            'total_votes': 0,
            'correct_votes': 0
        }
        self.stake_pool[validator_id] = stake
        return True

    def vote_on_proposal(
        self,
        proposal: Dict[str, Any],
        validator_id: str,
        vote: bool
    ) -> bool:
        """Vote on a proposal."""
        if validator_id not in self.validators:
            return False

        # Record vote
        self.validators[validator_id]['total_votes'] += 1
        if vote:
            self.validators[validator_id]['correct_votes'] += 1

        return True

    def get_consensus_result(self, proposal: Dict[str, Any]) -> Tuple[bool, float]:
        """Get consensus result for a proposal."""
        total_stake = sum(self.stake_pool.values())
        if total_stake == 0:
            return False, 0.0

        # Calculate weighted votes
        yes_stake = 0.0
        no_stake = 0.0 # Unused in ratio but good for logic

        for validator_id, validator_info in self.validators.items():
            # Simplified voting logic
            vote = validator_info['reputation'] > 0.5  # Simplified voting decision
            stake = validator_info['stake']

            if vote:
                yes_stake += stake
            else:
                no_stake += stake

        # Calculate consensus
        consensus_ratio = yes_stake / total_stake
        consensus_reached = consensus_ratio >= self.consensus_threshold

        return consensus_reached, consensus_ratio

    def update_validator_reputation(self, validator_id: str, correct: bool) -> None:
        """Update validator reputation based on voting accuracy."""
        if validator_id in self.validators:
            if correct:
                self.validators[validator_id]['reputation'] = min(1.0,
                    self.validators[validator_id]['reputation'] + 0.1)
            else:
                self.validators[validator_id]['reputation'] = max(0.0,
                    self.validators[validator_id]['reputation'] - 0.05)


@dataclass
class BlockchainRouterConfig(RouterConfig):
    """Configuration for blockchain-based router."""
    blockchain_enabled: bool = True
    smart_contract_address: str = "0x1234567890abcdef"
    consensus_mechanism: str = "proof_of_stake"  # proof_of_work, proof_of_stake, delegated_proof_of_stake
    verification_required: bool = True
    reputation_threshold: float = 0.5
    stake_required: float = 100.0
    consensus_threshold: float = 0.51
    block_time: float = 10.0  # seconds
    mining_reward: float = 10.0
    transaction_fee: float = 0.1
    enable_smart_contracts: bool = True
    enable_consensus: bool = True
    enable_reputation_system: bool = True
    enable_stake_system: bool = True
    max_validators: int = 100
    validator_timeout: float = 300.0  # seconds


class BlockchainRouter(BaseRouter):
    """
    Blockchain-based router using distributed ledger technology for expert verification and consensus.
    """

    def __init__(self, config: BlockchainRouterConfig) -> None:
        """
        Initialize BlockchainRouter.

        Args:
            config: Blockchain router configuration.
        """
        super().__init__(config)
        self.config = config
        self.blockchain: Optional[Blockchain] = None
        self.smart_contract: Optional[SmartContract] = None
        self.consensus_mechanism: Optional[ConsensusMechanism] = None
        self.node_address: Optional[str] = None
        self.private_key = None
        self.public_key = None
        self.routing_transactions: List[Dict[str, Any]] = []
        self.consensus_history: List[Any] = []

    def initialize(self) -> None:
        """Initialize the blockchain router."""
        # Create blockchain
        if self.config.blockchain_enabled:
            self.blockchain = Blockchain(difficulty=4)

        # Create smart contract
        if self.config.enable_smart_contracts:
            self.smart_contract = SmartContract(self.config.smart_contract_address)

        # Create consensus mechanism
        if self.config.enable_consensus:
            self.consensus_mechanism = ConsensusMechanism(self.config.consensus_mechanism)

        # Generate cryptographic keys
        self._generate_keys()

        # Initialize node
        self.node_address = self._generate_node_address()

        self._initialized = True
        logger.info(f"Blockchain router initialized with consensus: {self.config.consensus_mechanism}")

    def _generate_keys(self) -> None:
        """Generate cryptographic keys for the node."""
        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def _generate_node_address(self) -> str:
        """Generate a unique node address."""
        if self.public_key is None:
            raise ValueError("Keys not generated")

        public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_key_bytes).hexdigest()[:16]

    def route_tokens(
        self,
        input_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """
        Route tokens using blockchain-based consensus.

        Args:
            input_tokens: Input tokens tensor.
            attention_mask: Attention mask.
            context: Context dictionary.

        Returns:
            RoutingResult object.
        """
        start_time = time.time()

        # Validate input
        self.validate_input(input_tokens)

        # Check cache
        cache_key = self.get_cache_key(input_tokens, context)
        if cache_key:
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                return cached_result

        # Extract blockchain features
        blockchain_features = self._extract_blockchain_features(input_tokens, attention_mask, context)

        # Apply blockchain routing
        expert_indices, expert_weights, confidence = self._blockchain_routing(blockchain_features, context)

        # Record routing transaction
        self._record_routing_transaction(expert_indices, expert_weights, confidence, context)

        # Create routing result
        result = RoutingResult(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            routing_confidence=confidence,
            routing_time=time.time() - start_time,
            strategy_used="blockchain_consensus",
            metadata={
                'node_address': self.node_address,
                'consensus_mechanism': self.config.consensus_mechanism,
                'verification_required': self.config.verification_required,
                'reputation_threshold': self.config.reputation_threshold,
                'blockchain_enabled': self.config.blockchain_enabled,
                'smart_contract_enabled': self.config.enable_smart_contracts
            }
        )

        # Cache result
        if cache_key:
            self.cache_result(cache_key, result)

        # Record metrics and log
        self.record_metrics(result)
        self.log_routing(result, input_tokens.shape)

        return result

    def _extract_blockchain_features(
        self,
        input_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Extract blockchain-specific features."""
        # Unused args for features currently, but kept for interface
        _ = attention_mask
        _ = context

        batch_size, seq_len, hidden_size = input_tokens.shape

        features: List[float] = []

        # Blockchain state features
        if self.blockchain and self.node_address:
            features.extend([
                float(len(self.blockchain.chain)),
                float(len(self.blockchain.pending_transactions)),
                self.blockchain.get_balance(self.node_address)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Smart contract features
        if self.smart_contract:
            features.extend([
                float(len(self.smart_contract.experts)),
                float(len(self.smart_contract.routing_history)),
                self.smart_contract.consensus_threshold
            ])
        else:
            features.extend([0.0, 0.0, 0.5])

        # Consensus features
        if self.consensus_mechanism:
            features.extend([
                float(len(self.consensus_mechanism.validators)),
                sum(self.consensus_mechanism.stake_pool.values()),
                self.consensus_mechanism.consensus_threshold
            ])
        else:
            features.extend([0.0, 0.0, 0.51])

        # Pad or truncate to hidden_size
        while len(features) < self.config.hidden_size:
            features.append(0.0)
        features = features[:self.config.hidden_size]

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _blockchain_routing(
        self,
        blockchain_features: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[int], List[float], float]:
        """Perform blockchain-based routing."""
        if self.smart_contract:
            # Use smart contract for routing
            request_data = {
                'request_id': context.get('request_id', '') if context else '',
                'required_capabilities': context.get('required_capabilities', []) if context else [],
                'timestamp': time.time()
            }

            routing_result = self.smart_contract.route_request(request_data)

            if 'error' in routing_result:
                # Fallback to random routing
                expert_indices = [random.randint(0, self.config.num_experts - 1)]
                expert_weights = [1.0]
                confidence = 0.5
            else:
                # Assuming expert_id in smart contract maps to integer index or we interpret it
                # Logic gap here if expert_id is string "expert_1", we need int.
                # Assuming simple conversion for demo:
                expert_id_str = routing_result.get('expert_id', '0')
                try:
                    # Attempt to parse int if format "expert_<ID>" or just "<ID>"
                    if expert_id_str.startswith('expert_'):
                         idx = int(expert_id_str.split('_')[1])
                    else:
                         idx = int(expert_id_str)
                except (ValueError, IndexError):
                    idx = 0

                expert_indices = [idx]
                expert_weights = [float(routing_result.get('confidence', 0.5))]
                confidence = float(routing_result.get('confidence', 0.5))
        else:
            # Fallback to consensus-based routing
            expert_indices, expert_weights, confidence = self._consensus_routing(blockchain_features)

        return expert_indices, expert_weights, confidence

    def _consensus_routing(self, blockchain_features: torch.Tensor) -> Tuple[List[int], List[float], float]:
        """Perform consensus-based routing."""
        if not self.consensus_mechanism:
            # Fallback to random routing
            expert_indices = [random.randint(0, self.config.num_experts - 1)]
            expert_weights = [1.0]
            confidence = 0.5
        else:
            # Create routing proposal
            proposal = {
                'type': 'routing_decision',
                'timestamp': time.time(),
                'features': blockchain_features.cpu().numpy().tolist()
            }

            # Get consensus on proposal
            consensus_reached, consensus_ratio = self.consensus_mechanism.get_consensus_result(proposal)

            if consensus_reached:
                # Select expert based on consensus (Simplified to random since consensus doesn't output expert ID here)
                expert_indices = [random.randint(0, self.config.num_experts - 1)]
                expert_weights = [consensus_ratio]
                confidence = consensus_ratio
            else:
                # Fallback to random routing
                expert_indices = [random.randint(0, self.config.num_experts - 1)]
                expert_weights = [1.0]
                confidence = 0.5

        return expert_indices, expert_weights, confidence

    def _record_routing_transaction(
        self,
        expert_indices: List[int],
        expert_weights: List[float],
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record routing transaction on blockchain."""
        if not self.blockchain or not self.node_address:
            return

        # Create routing transaction
        transaction = {
            'from': self.node_address,
            'to': f'expert_{expert_indices[0]}' if expert_indices else 'unknown',
            'amount': confidence,
            'timestamp': time.time(),
            'signature': self._sign_transaction(expert_indices, expert_weights, confidence),
            'routing_data': {
                'expert_indices': expert_indices,
                'expert_weights': expert_weights,
                'confidence': confidence
            }
        }

        # Add to blockchain
        self.blockchain.add_transaction(transaction)

        # Record in local history
        self.routing_transactions.append(transaction)

    def _sign_transaction(self, expert_indices: List[int], expert_weights: List[float], confidence: float) -> str:
        """Sign a transaction."""
        if self.private_key is None:
             return ""

        # Create transaction data
        transaction_data = {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'confidence': confidence,
            'timestamp': time.time()
        }

        # Sign with private key
        message = json.dumps(transaction_data, sort_keys=True).encode()
        try:
            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"Failed to sign transaction: {e}")
            return ""

    def register_expert(self, expert_id: str, expert_data: Dict[str, Any]) -> bool:
        """Register an expert on the blockchain."""
        if not self.smart_contract:
            return False

        # Sign expert data
        signature = self._sign_transaction([], [], 0.0)  # Simplified signature

        # Register expert
        return self.smart_contract.register_expert(expert_id, expert_data, signature)

    def verify_expert(self, expert_id: str, verification_data: Dict[str, Any]) -> bool:
        """Verify an expert."""
        if not self.smart_contract:
            return False

        return self.smart_contract.verify_expert(expert_id, verification_data)

    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics."""
        stats = {
            'blockchain_enabled': self.config.blockchain_enabled,
            'smart_contract_enabled': self.config.enable_smart_contracts,
            'consensus_enabled': self.config.enable_consensus,
            'node_address': self.node_address,
            'consensus_mechanism': self.config.consensus_mechanism
        }

        if self.blockchain and self.node_address:
            stats.update({
                'chain_length': len(self.blockchain.chain),
                'pending_transactions': len(self.blockchain.pending_transactions),
                'node_balance': self.blockchain.get_balance(self.node_address),
                'chain_valid': self.blockchain.validate_chain()
            })

        if self.smart_contract:
            stats.update({
                'registered_experts': len(self.smart_contract.experts),
                'routing_history': len(self.smart_contract.routing_history)
            })

        if self.consensus_mechanism:
            stats.update({
                'validators': len(self.consensus_mechanism.validators),
                'total_stake': sum(self.consensus_mechanism.stake_pool.values()),
                'consensus_threshold': self.consensus_mechanism.consensus_threshold
            })

        return stats

    def get_router_info(self) -> Dict[str, Any]:
        """Get router information and statistics."""
        base_info = super().get_router_info()
        base_info.update({
            'router_type': 'blockchain_based',
            'blockchain_enabled': self.config.blockchain_enabled,
            'smart_contract_address': self.config.smart_contract_address,
            'consensus_mechanism': self.config.consensus_mechanism,
            'verification_required': self.config.verification_required,
            'reputation_threshold': self.config.reputation_threshold,
            'stake_required': self.config.stake_required,
            'consensus_threshold': self.config.consensus_threshold,
            'blockchain_stats': self.get_blockchain_stats()
        })
        return base_info

