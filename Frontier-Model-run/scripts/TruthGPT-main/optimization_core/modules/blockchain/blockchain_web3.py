"""
Blockchain & Web3 Integration Module for TruthGPT Optimization Core
Implements decentralized model versioning, federated learning, and IPFS storage
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import json
import pickle
import hashlib
from collections import defaultdict, deque
import math
import random
from pathlib import Path
import asyncio
from contextlib import contextmanager
import os
import requests
import base64

logger = logging.getLogger(__name__)

class BlockchainType(Enum):
    """Supported blockchain types"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    AVALANCHE = "avalanche"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"
    POLKADOT = "polkadot"

class SmartContractType(Enum):
    """Types of smart contracts"""
    MODEL_REGISTRY = "model_registry"
    FEDERATED_LEARNING = "federated_learning"
    DATA_MARKETPLACE = "data_marketplace"
    REWARD_DISTRIBUTION = "reward_distribution"
    GOVERNANCE = "governance"
    STAKING = "staking"

class ConsensusMechanism(Enum):
    """Consensus mechanisms"""
    PROOF_OF_WORK = "pow"
    PROOF_OF_STAKE = "pos"
    PROOF_OF_AUTHORITY = "poa"
    DELEGATED_PROOF_OF_STAKE = "dpos"
    PROOF_OF_HISTORY = "poh"

@dataclass
class BlockchainConfig:
    """Configuration for blockchain integration"""
    # Blockchain settings
    blockchain_type: BlockchainType = BlockchainType.ETHEREUM
    network: str = "mainnet"  # mainnet, testnet, local
    rpc_url: str = ""
    private_key: str = ""
    contract_address: str = ""
    
    # Gas settings
    gas_limit: int = 300000
    gas_price: int = 20  # gwei
    max_fee_per_gas: int = 50  # gwei
    max_priority_fee_per_gas: int = 2  # gwei
    
    # Transaction settings
    timeout: int = 300  # seconds
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    
    # IPFS settings
    ipfs_gateway: str = "https://ipfs.io/ipfs/"
    ipfs_api_url: str = "http://localhost:5001"
    
    # Model settings
    model_compression: bool = True
    encryption_enabled: bool = True
    encryption_key: str = ""
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.rpc_url:
            raise ValueError("RPC URL is required")
        if not self.private_key and self.network != "local":
            raise ValueError("Private key is required for non-local networks")

@dataclass
class ModelMetadata:
    """Metadata for blockchain-stored models"""
    model_id: str
    version: str
    hash: str
    size: int
    architecture: str
    training_data_hash: str
    performance_metrics: Dict[str, float]
    timestamp: float
    creator: str
    license: str
    ipfs_hash: str
    blockchain_tx_hash: str

class BlockchainConnector:
    """Connector for blockchain interactions"""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Blockchain connection
        self.web3 = None
        self.account = None
        self.contracts = {}
        
        # Initialize connection
        self._initialize_connection()
        
        logger.info(f"✅ Blockchain Connector initialized for {config.blockchain_type.value}")
    
    def _initialize_connection(self):
        """Initialize blockchain connection"""
        try:
            # Import web3
            from web3 import Web3
            
            # Connect to blockchain
            self.web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            
            if not self.web3.is_connected():
                raise ConnectionError("Failed to connect to blockchain")
            
            # Setup account
            if self.config.private_key:
                self.account = self.web3.eth.account.from_key(self.config.private_key)
            
            logger.info("✅ Blockchain connection established")
            
        except ImportError:
            logger.warning("Web3 not available, using mock implementation")
            self._setup_mock_connection()
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connection: {e}")
            self._setup_mock_connection()
    
    def _setup_mock_connection(self):
        """Setup mock blockchain connection for testing"""
        self.web3 = MockWeb3()
        self.account = MockAccount()
        logger.info("✅ Mock blockchain connection established")
    
    def get_balance(self) -> float:
        """Get account balance"""
        if not self.account:
            return 0.0
        
        try:
            balance_wei = self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balance_wei, 'ether')
            return float(balance_eth)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def send_transaction(self, to_address: str, value: float, data: str = "") -> str:
        """Send a transaction"""
        if not self.account:
            raise ValueError("No account configured")
        
        try:
            # Build transaction
            transaction = {
                'from': self.account.address,
                'to': to_address,
                'value': self.web3.to_wei(value, 'ether'),
                'gas': self.config.gas_limit,
                'gasPrice': self.web3.to_wei(self.config.gas_price, 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'data': data
            }
        
            # Sign transaction
            signed_txn = self.web3.eth.account.sign_transaction(transaction, self.account.key)
        
            # Send transaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return tx_hash.hex()
    
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            return ""
    
    def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction receipt"""
        try:
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            return dict(receipt)
        except Exception as e:
            logger.error(f"Failed to get transaction receipt: {e}")
            return {}

class SmartContractManager:
    """Manager for smart contract interactions"""
    
    def __init__(self, config: BlockchainConfig, connector: BlockchainConnector):
        self.config = config
        self.connector = connector
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Contract instances
        self.contracts = {}
        
        # Initialize contracts
        self._initialize_contracts()
        
        logger.info("✅ Smart Contract Manager initialized")
    
    def _initialize_contracts(self):
        """Initialize smart contracts"""
        # Model Registry Contract
        self.contracts['model_registry'] = ModelRegistryContract(self.config, self.connector)
        
        # Federated Learning Contract
        self.contracts['federated_learning'] = FederatedLearningContract(self.config, self.connector)
        
        logger.info("✅ Smart contracts initialized")
    
    def get_contract(self, contract_type: SmartContractType):
        """Get contract instance"""
        return self.contracts.get(contract_type.value)

class ModelRegistryContract:
    """Smart contract for model registry"""
    
    def __init__(self, config: BlockchainConfig, connector: BlockchainConnector):
        self.config = config
        self.connector = connector
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Contract ABI (simplified)
        self.abi = [
            {
                "inputs": [
                    {"name": "modelId", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "hash", "type": "string"},
                    {"name": "ipfsHash", "type": "string"}
                ],
                "name": "registerModel",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "modelId", "type": "string"}],
                "name": "getModel",
                "outputs": [
                    {"name": "version", "type": "string"},
                    {"name": "hash", "type": "string"},
                    {"name": "ipfsHash", "type": "string"},
                    {"name": "timestamp", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        logger.info("✅ Model Registry Contract initialized")
    
    def register_model(self, model_metadata: ModelMetadata) -> str:
        """Register model on blockchain"""
        try:
            # Prepare transaction data
            data = self._encode_register_model(model_metadata)
        
        # Send transaction
            tx_hash = self.connector.send_transaction(
                to_address=self.config.contract_address,
                value=0,
                data=data
            )
            
            if tx_hash:
                logger.info(f"✅ Model registered: {model_metadata.model_id}")
                return tx_hash
            else:
                logger.error("Failed to register model")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return ""
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get model from blockchain"""
        try:
            # Call contract function
            result = self._call_contract_function("getModel", [model_id])
            return result
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return {}
    
    def _encode_register_model(self, metadata: ModelMetadata) -> str:
        """Encode register model function call"""
        # Simplified encoding
        return f"0x{hashlib.sha256(f'registerModel{metadata.model_id}{metadata.version}'.encode()).hexdigest()[:8]}"
    
    def _call_contract_function(self, function_name: str, args: List[Any]) -> Dict[str, Any]:
        """Call contract function"""
        # Mock implementation
        return {
            'version': '1.0.0',
            'hash': '0x1234567890abcdef',
            'ipfsHash': 'QmMockHash',
            'timestamp': int(time.time())
        }

class FederatedLearningContract:
    """Smart contract for federated learning coordination"""
    
    def __init__(self, config: BlockchainConfig, connector: BlockchainConnector):
        self.config = config
        self.connector = connector
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Federated learning state
        self.rounds = {}
        self.participants = {}
        self.aggregations = {}
        
        logger.info("✅ Federated Learning Contract initialized")
    
    def start_federated_round(self, round_id: str, model_hash: str) -> str:
        """Start a new federated learning round"""
        try:
            self.rounds[round_id] = {
                'id': round_id,
                'model_hash': model_hash,
                'start_time': time.time(),
                'participants': [],
                'status': 'active'
            }
            
            logger.info(f"✅ Federated round started: {round_id}")
            return round_id
            
        except Exception as e:
            logger.error(f"Failed to start federated round: {e}")
            return ""
    
    def join_round(self, round_id: str, participant_id: str, model_hash: str) -> bool:
        """Join a federated learning round"""
        try:
            if round_id not in self.rounds:
                logger.error(f"Round {round_id} not found")
                return False
            
            self.rounds[round_id]['participants'].append({
                'id': participant_id,
                'model_hash': model_hash,
                'join_time': time.time()
            })
            
            logger.info(f"✅ Participant {participant_id} joined round {round_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join round: {e}")
            return False
    
    def submit_gradient(self, round_id: str, participant_id: str, gradient_hash: str) -> bool:
        """Submit gradient for aggregation"""
        try:
            if round_id not in self.rounds:
                logger.error(f"Round {round_id} not found")
                return False
            
            # Find participant
            for participant in self.rounds[round_id]['participants']:
                if participant['id'] == participant_id:
                    participant['gradient_hash'] = gradient_hash
                    participant['submit_time'] = time.time()
                    break
            
            logger.info(f"✅ Gradient submitted by {participant_id} for round {round_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit gradient: {e}")
            return False
    
    def aggregate_gradients(self, round_id: str) -> str:
        """Aggregate gradients from all participants"""
        try:
            if round_id not in self.rounds:
                logger.error(f"Round {round_id} not found")
                return ""
            
            round_data = self.rounds[round_id]
            participants = round_data['participants']
            
            # Check if all participants have submitted gradients
            submitted_count = sum(1 for p in participants if 'gradient_hash' in p)
            
            if submitted_count < len(participants):
                logger.warning(f"Not all participants have submitted gradients: {submitted_count}/{len(participants)}")
                return ""
            
            # Aggregate gradients (simplified)
            aggregated_hash = self._aggregate_gradient_hashes(participants)
            
            # Store aggregation
            self.aggregations[round_id] = {
                'round_id': round_id,
                'aggregated_hash': aggregated_hash,
                'participant_count': len(participants),
                'aggregation_time': time.time()
            }
            
            # Update round status
            round_data['status'] = 'completed'
            round_data['aggregated_hash'] = aggregated_hash
            
            logger.info(f"✅ Gradients aggregated for round {round_id}")
            return aggregated_hash
            
        except Exception as e:
            logger.error(f"Failed to aggregate gradients: {e}")
            return ""
    
    def _aggregate_gradient_hashes(self, participants: List[Dict[str, Any]]) -> str:
        """Aggregate gradient hashes"""
        # Simplified aggregation - in practice, this would involve actual gradient computation
        combined_hash = ""
        for participant in participants:
            if 'gradient_hash' in participant:
                combined_hash += participant['gradient_hash']
        
        return hashlib.sha256(combined_hash.encode()).hexdigest()
    
    def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """Get federated learning round status"""
        if round_id not in self.rounds:
            return {}
        
        round_data = self.rounds[round_id]
        return {
            'id': round_data['id'],
            'status': round_data['status'],
            'participant_count': len(round_data['participants']),
            'submitted_count': sum(1 for p in round_data['participants'] if 'gradient_hash' in p),
            'start_time': round_data['start_time'],
            'aggregated_hash': round_data.get('aggregated_hash', '')
        }

class IPFSManager:
    """Manager for IPFS interactions"""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # IPFS connection
        self.api_url = config.ipfs_api_url
        self.gateway_url = config.ipfs_gateway
        
        logger.info("✅ IPFS Manager initialized")
    
    def upload_model(self, model: nn.Module, metadata: Dict[str, Any]) -> str:
        """Upload model to IPFS"""
        try:
            # Serialize model
            model_data = self._serialize_model(model)
            
            # Create IPFS object
            ipfs_object = {
                'model': model_data,
                'metadata': metadata,
                'timestamp': time.time()
            }
            
            # Upload to IPFS
            ipfs_hash = self._upload_to_ipfs(ipfs_object)
            
            if ipfs_hash:
                logger.info(f"✅ Model uploaded to IPFS: {ipfs_hash}")
                return ipfs_hash
            else:
                logger.error("Failed to upload model to IPFS")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to upload model to IPFS: {e}")
            return ""
    
    def download_model(self, ipfs_hash: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """Download model from IPFS"""
        try:
            # Download from IPFS
            ipfs_object = self._download_from_ipfs(ipfs_hash)
            
            if not ipfs_object:
                logger.error("Failed to download from IPFS")
                return None, {}
            
            # Deserialize model
            model = self._deserialize_model(ipfs_object['model'])
            metadata = ipfs_object['metadata']
            
            logger.info(f"✅ Model downloaded from IPFS: {ipfs_hash}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to download model from IPFS: {e}")
            return None, {}
    
    def _serialize_model(self, model: nn.Module) -> bytes:
        """Serialize model for IPFS storage"""
        # Save model to bytes
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()
    
    def _deserialize_model(self, model_data: bytes) -> nn.Module:
        """Deserialize model from IPFS data"""
        # Load model from bytes
        buffer = io.BytesIO(model_data)
        state_dict = torch.load(buffer, map_location='cpu')
        
        # Create model (simplified)
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        model.load_state_dict(state_dict)
        return model
    
    def _upload_to_ipfs(self, data: Dict[str, Any]) -> str:
        """Upload data to IPFS"""
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Upload to IPFS API
            files = {'file': ('data.pkl', serialized_data)}
            response = requests.post(f"{self.api_url}/api/v0/add", files=files)
            
            if response.status_code == 200:
                result = response.json()
                return result['Hash']
            else:
                logger.error(f"IPFS upload failed: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to upload to IPFS: {e}")
            return ""
    
    def _download_from_ipfs(self, ipfs_hash: str) -> Optional[Dict[str, Any]]:
        """Download data from IPFS"""
        try:
            # Download from IPFS gateway
            response = requests.get(f"{self.gateway_url}{ipfs_hash}")
            
            if response.status_code == 200:
                return pickle.loads(response.content)
            else:
                logger.error(f"IPFS download failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download from IPFS: {e}")
            return None

class TruthGPTBlockchainManager:
    """Main blockchain manager coordinating all components"""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.connector = BlockchainConnector(config)
        self.contract_manager = SmartContractManager(config, self.connector)
        self.ipfs_manager = IPFSManager(config)
        
        # Blockchain state
        self.registered_models = {}
        self.federated_rounds = {}
        
        logger.info("✅ TruthGPT Blockchain Manager initialized")
    
    def register_model(self, model: nn.Module, metadata: Dict[str, Any]) -> str:
        """Register model on blockchain"""
        try:
            # Generate model ID
            model_id = self._generate_model_id(model, metadata)
        
        # Calculate model hash
            model_hash = self._calculate_model_hash(model)
                
                # Upload to IPFS
            ipfs_hash = self.ipfs_manager.upload_model(model, metadata)
            
            if not ipfs_hash:
                logger.error("Failed to upload model to IPFS")
                return ""
        
        # Create model metadata
        model_metadata = ModelMetadata(
                model_id=model_id,
            version=metadata.get('version', '1.0.0'),
                hash=model_hash,
                size=self._calculate_model_size(model),
                architecture=metadata.get('architecture', 'unknown'),
                training_data_hash=metadata.get('training_data_hash', ''),
                performance_metrics=metadata.get('performance_metrics', {}),
                timestamp=time.time(),
                creator=metadata.get('creator', 'unknown'),
            license=metadata.get('license', 'MIT'),
                ipfs_hash=ipfs_hash,
                blockchain_tx_hash=''
        )
        
        # Register on blockchain
            model_registry = self.contract_manager.get_contract(SmartContractType.MODEL_REGISTRY)
            tx_hash = model_registry.register_model(model_metadata)
            
            if tx_hash:
                model_metadata.blockchain_tx_hash = tx_hash
                self.registered_models[model_id] = model_metadata
                logger.info(f"✅ Model registered: {model_id}")
                return model_id
            else:
                logger.error("Failed to register model on blockchain")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return ""
    
    def download_model(self, model_id: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """Download model from blockchain and IPFS"""
        try:
            # Get model metadata from blockchain
            model_registry = self.contract_manager.get_contract(SmartContractType.MODEL_REGISTRY)
            blockchain_data = model_registry.get_model(model_id)
            
            if not blockchain_data:
                logger.error(f"Model {model_id} not found on blockchain")
                return None, {}
            
            # Download from IPFS
            ipfs_hash = blockchain_data['ipfsHash']
            model, metadata = self.ipfs_manager.download_model(ipfs_hash)
            
            if model:
                logger.info(f"✅ Model downloaded: {model_id}")
                return model, metadata
            else:
                logger.error(f"Failed to download model {model_id} from IPFS")
                return None, {}
                
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return None, {}
    
    def start_federated_learning(self, model_id: str, participants: List[str]) -> str:
        """Start federated learning round"""
        try:
            # Generate round ID
            round_id = f"round_{int(time.time())}"
            
            # Get model hash
            if model_id in self.registered_models:
                model_hash = self.registered_models[model_id].hash
            else:
                model_hash = self._generate_model_hash()
            
            # Start federated round
            federated_contract = self.contract_manager.get_contract(SmartContractType.FEDERATED_LEARNING)
            round_id = federated_contract.start_federated_round(round_id, model_hash)
            
            if round_id:
                self.federated_rounds[round_id] = {
                    'model_id': model_id,
                    'participants': participants,
                    'start_time': time.time(),
                    'status': 'active'
                }
                logger.info(f"✅ Federated learning started: {round_id}")
                return round_id
            else:
                logger.error("Failed to start federated learning")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to start federated learning: {e}")
            return ""
    
    def join_federated_round(self, round_id: str, participant_id: str, model: nn.Module) -> bool:
        """Join federated learning round"""
        try:
            # Calculate model hash
            model_hash = self._calculate_model_hash(model)
            
            # Join round
            federated_contract = self.contract_manager.get_contract(SmartContractType.FEDERATED_LEARNING)
            success = federated_contract.join_round(round_id, participant_id, model_hash)
            
            if success:
                logger.info(f"✅ Participant {participant_id} joined round {round_id}")
                return True
            else:
                logger.error(f"Failed to join round {round_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to join federated round: {e}")
            return False
    
    def submit_federated_gradient(self, round_id: str, participant_id: str, gradient: torch.Tensor) -> bool:
        """Submit gradient for federated learning"""
        try:
            # Calculate gradient hash
            gradient_hash = self._calculate_tensor_hash(gradient)
            
            # Submit gradient
            federated_contract = self.contract_manager.get_contract(SmartContractType.FEDERATED_LEARNING)
            success = federated_contract.submit_gradient(round_id, participant_id, gradient_hash)
            
            if success:
                logger.info(f"✅ Gradient submitted by {participant_id} for round {round_id}")
                return True
            else:
                logger.error(f"Failed to submit gradient for round {round_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to submit federated gradient: {e}")
            return False
    
    def get_federated_round_status(self, round_id: str) -> Dict[str, Any]:
        """Get federated learning round status"""
        try:
            federated_contract = self.contract_manager.get_contract(SmartContractType.FEDERATED_LEARNING)
            return federated_contract.get_round_status(round_id)
        except Exception as e:
            logger.error(f"Failed to get round status: {e}")
            return {}
    
    def _generate_model_id(self, model: nn.Module, metadata: Dict[str, Any]) -> str:
        """Generate unique model ID"""
        model_str = str(model.state_dict())
        metadata_str = json.dumps(metadata, sort_keys=True)
        combined = f"{model_str}{metadata_str}{time.time()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Calculate model hash"""
        state_dict = model.state_dict()
        state_str = json.dumps({k: v.tolist() for k, v in state_dict.items()}, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes"""
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 4  # Assuming float32
    
    def _calculate_tensor_hash(self, tensor: torch.Tensor) -> str:
        """Calculate tensor hash"""
        tensor_str = tensor.detach().cpu().numpy().tobytes()
        return hashlib.sha256(tensor_str).hexdigest()
    
    def _generate_model_hash(self) -> str:
        """Generate random model hash for testing"""
        return hashlib.sha256(f"model_{time.time()}".encode()).hexdigest()
    
    def get_blockchain_summary(self) -> Dict[str, Any]:
        """Get blockchain integration summary"""
        return {
            'blockchain_type': self.config.blockchain_type.value,
            'network': self.config.network,
            'account_balance': self.connector.get_balance(),
            'registered_models': len(self.registered_models),
            'active_federated_rounds': len(self.federated_rounds),
            'ipfs_gateway': self.config.ipfs_gateway
        }

# Mock implementations for testing
class MockWeb3:
    """Mock Web3 implementation for testing"""
    
    def __init__(self):
        self.eth = MockEth()
    
    def is_connected(self):
        return True
    
    def from_wei(self, value, unit):
        return value / (10**18)
    
    def to_wei(self, value, unit):
        return int(value * (10**18))

class MockEth:
    """Mock Ethereum implementation"""
    
    def get_balance(self, address):
        return 1000000000000000000  # 1 ETH in wei
    
    def get_transaction_count(self, address):
        return 0

class MockAccount:
    """Mock account implementation"""
    
    def __init__(self):
        self.address = "0x1234567890123456789012345678901234567890"
        self.key = "mock_private_key"

# Factory functions
def create_blockchain_config(**kwargs) -> BlockchainConfig:
    """Create blockchain configuration"""
    return BlockchainConfig(**kwargs)

def create_blockchain_connector(config: BlockchainConfig) -> BlockchainConnector:
    """Create blockchain connector"""
    return BlockchainConnector(config)

def create_ipfs_manager(config: BlockchainConfig) -> IPFSManager:
    """Create IPFS manager"""
    return IPFSManager(config)

def create_model_registry_contract(config: BlockchainConfig, connector: BlockchainConnector) -> ModelRegistryContract:
    """Create model registry contract"""
    return ModelRegistryContract(config, connector)

def create_federated_learning_contract(config: BlockchainConfig, connector: BlockchainConnector) -> FederatedLearningContract:
    """Create federated learning contract"""
    return FederatedLearningContract(config, connector)

def create_blockchain_manager(config: BlockchainConfig) -> TruthGPTBlockchainManager:
    """Create blockchain manager"""
    return TruthGPTBlockchainManager(config)

# Example usage
def example_blockchain_integration():
    """Example of blockchain integration"""
    # Create configuration
    config = create_blockchain_config(
        blockchain_type=BlockchainType.ETHEREUM,
        network="testnet",
        rpc_url="https://goerli.infura.io/v3/YOUR_PROJECT_ID",
        contract_address="0x1234567890123456789012345678901234567890"
    )
    
    # Create blockchain manager
    manager = create_blockchain_manager(config)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Register model
    metadata = {
        'version': '1.0.0',
        'architecture': 'simple_mlp',
        'performance_metrics': {'accuracy': 0.95, 'loss': 0.1},
        'creator': 'TruthGPT',
        'license': 'MIT'
    }
    
    model_id = manager.register_model(model, metadata)
    
    if model_id:
        print(f"✅ Model registered: {model_id}")
        
        # Download model
        downloaded_model, downloaded_metadata = manager.download_model(model_id)
        
        if downloaded_model:
            print("✅ Model downloaded successfully")
        
        # Start federated learning
        participants = ["participant1", "participant2", "participant3"]
        round_id = manager.start_federated_learning(model_id, participants)
        
        if round_id:
            print(f"✅ Federated learning started: {round_id}")
            
            # Join round
            success = manager.join_federated_round(round_id, "participant1", model)
            if success:
                print("✅ Participant joined federated round")
            
            # Submit gradient
            gradient = torch.randn(10, 50)
            success = manager.submit_federated_gradient(round_id, "participant1", gradient)
            if success:
                print("✅ Gradient submitted")
            
            # Get round status
            status = manager.get_federated_round_status(round_id)
            print(f"📊 Round status: {status}")
    
    # Get blockchain summary
    summary = manager.get_blockchain_summary()
    print(f"📊 Blockchain Summary: {summary}")
    
    return manager

# Export utilities
__all__ = [
    'BlockchainType',
    'SmartContractType',
    'ConsensusMechanism',
    'BlockchainConfig',
    'ModelMetadata',
    'BlockchainConnector',
    'SmartContractManager',
    'ModelRegistryContract',
    'FederatedLearningContract',
    'IPFSManager',
    'TruthGPTBlockchainManager',
    'create_blockchain_config',
    'create_blockchain_connector',
    'create_ipfs_manager',
    'create_model_registry_contract',
    'create_federated_learning_contract',
    'create_blockchain_manager',
    'example_blockchain_integration'
]

if __name__ == "__main__":
    import io
    example_blockchain_integration()
    print("✅ Blockchain integration module complete!")
