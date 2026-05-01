"""
TruthGPT Blockchain & Web3 Integration
Decentralized model management, smart contracts, and federated learning on blockchain
"""

import asyncio
import json
import time
import hashlib
import base64
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import pickle
import threading
from datetime import datetime, timedelta
import uuid
import requests
import aiohttp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ipfshttpclient
import web3
from web3 import Web3
from eth_account import Account
import solana
from solana.rpc.api import Client
from solana.publickey import PublicKey
from solana.keypair import Keypair

# Import TruthGPT modules
from .models import TruthGPTModel, TruthGPTModelConfig
from .ai_enhancement import TruthGPTAIEnhancementManager


class BlockchainType(Enum):
    """Supported blockchain types"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BINANCE_SMART_CHAIN = "bsc"
    SOLANA = "solana"
    AVALANCHE = "avalanche"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON_ZKEVM = "polygon_zkevm"


class SmartContractType(Enum):
    """Types of smart contracts"""
    MODEL_REGISTRY = "model_registry"
    FEDERATED_LEARNING = "federated_learning"
    MODEL_MARKETPLACE = "model_marketplace"
    DATA_MARKETPLACE = "data_marketplace"
    GOVERNANCE = "governance"
    STAKING = "staking"
    REWARDS = "rewards"


class ConsensusMechanism(Enum):
    """Consensus mechanisms"""
    PROOF_OF_WORK = "pow"
    PROOF_OF_STAKE = "pos"
    PROOF_OF_AUTHORITY = "poa"
    DELEGATED_PROOF_OF_STAKE = "dpos"
    PROOF_OF_HISTORY = "poh"
    PROOF_OF_SPACE = "pospace"


@dataclass
class BlockchainConfig:
    """Configuration for blockchain integration"""
    blockchain_type: BlockchainType = BlockchainType.ETHEREUM
    rpc_url: str = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
    private_key: Optional[str] = None
    contract_address: Optional[str] = None
    gas_limit: int = 200000
    gas_price: int = 20  # gwei
    enable_ipfs: bool = True
    ipfs_gateway: str = "https://ipfs.io/ipfs/"
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    enable_federated_learning: bool = True
    federation_nodes: List[str] = field(default_factory=list)
    consensus_threshold: float = 0.6
    enable_staking: bool = True
    staking_amount: int = 1000  # tokens


@dataclass
class ModelMetadata:
    """Model metadata for blockchain storage"""
    model_id: str
    version_id: str
    model_hash: str
    ipfs_hash: str
    creator: str
    created_at: int
    model_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_data_hash: str = ""
    license: str = "MIT"
    price: int = 0  # in tokens
    is_public: bool = True
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


class BlockchainConnector:
    """Blockchain connector for TruthGPT"""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.logger = logging.getLogger(f"BlockchainConnector_{id(self)}")
        
        # Initialize blockchain connection
        self.w3 = None
        self.account = None
        self._init_blockchain()
        
        # Initialize IPFS
        self.ipfs_client = None
        if config.enable_ipfs:
            self._init_ipfs()
        
        # Initialize encryption
        self.encryption_key = None
        if config.enable_encryption:
            self._init_encryption()
    
    def _init_blockchain(self):
        """Initialize blockchain connection"""
        try:
            if self.config.blockchain_type in [BlockchainType.ETHEREUM, BlockchainType.POLYGON, 
                                             BlockchainType.BINANCE_SMART_CHAIN, BlockchainType.ARBITRUM, 
                                             BlockchainType.OPTIMISM, BlockchainType.POLYGON_ZKEVM]:
                self.w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
                
                if self.config.private_key:
                    self.account = Account.from_key(self.config.private_key)
                    self.logger.info(f"Connected to {self.config.blockchain_type.value} with account {self.account.address}")
                else:
                    self.logger.warning("No private key provided, read-only mode")
            
            elif self.config.blockchain_type == BlockchainType.SOLANA:
                # Initialize Solana client
                self.solana_client = Client(self.config.rpc_url)
                self.logger.info(f"Connected to Solana network")
            
            else:
                self.logger.error(f"Unsupported blockchain type: {self.config.blockchain_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize blockchain connection: {e}")
    
    def _init_ipfs(self):
        """Initialize IPFS connection"""
        try:
            self.ipfs_client = ipfshttpclient.connect()
            self.logger.info("Connected to IPFS")
        except Exception as e:
            self.logger.warning(f"Failed to connect to IPFS: {e}")
            self.ipfs_client = None
    
    def _init_encryption(self):
        """Initialize encryption"""
        if self.config.encryption_key:
            self.encryption_key = self.config.encryption_key.encode()
        else:
            # Generate encryption key
            password = b"truthgpt_default_password"
            salt = b"truthgpt_salt"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data"""
        if not self.encryption_key:
            return data
        
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        if not self.encryption_key:
            return encrypted_data
        
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_data)
    
    async def upload_to_ipfs(self, data: bytes, encrypt: bool = True) -> str:
        """Upload data to IPFS"""
        if not self.ipfs_client:
            raise Exception("IPFS client not initialized")
        
        # Encrypt data if requested
        if encrypt:
            data = self.encrypt_data(data)
        
        # Upload to IPFS
        result = self.ipfs_client.add_bytes(data)
        ipfs_hash = result['Hash']
        
        self.logger.info(f"Uploaded data to IPFS: {ipfs_hash}")
        return ipfs_hash
    
    async def download_from_ipfs(self, ipfs_hash: str, decrypt: bool = True) -> bytes:
        """Download data from IPFS"""
        if not self.ipfs_client:
            raise Exception("IPFS client not initialized")
        
        # Download from IPFS
        data = self.ipfs_client.cat(ipfs_hash)
        
        # Decrypt data if requested
        if decrypt:
            data = self.decrypt_data(data)
        
        return data
    
    def calculate_model_hash(self, model_data: bytes) -> str:
        """Calculate hash of model data"""
        return hashlib.sha256(model_data).hexdigest()
    
    async def deploy_smart_contract(self, contract_type: SmartContractType, 
                                  contract_bytecode: str, constructor_args: List[Any] = None) -> str:
        """Deploy smart contract"""
        if not self.w3 or not self.account:
            raise Exception("Blockchain connection not initialized")
        
        if self.config.blockchain_type == BlockchainType.SOLANA:
            return await self._deploy_solana_contract(contract_type, contract_bytecode, constructor_args)
        else:
            return await self._deploy_ethereum_contract(contract_type, contract_bytecode, constructor_args)
    
    async def _deploy_ethereum_contract(self, contract_type: SmartContractType, 
                                      contract_bytecode: str, constructor_args: List[Any] = None) -> str:
        """Deploy Ethereum-compatible smart contract"""
        # This is a simplified deployment - in production, you'd use proper contract deployment
        contract_address = f"0x{hashlib.sha256(f'{contract_type.value}_{time.time()}'.encode()).hexdigest()[:40]}"
        
        self.logger.info(f"Deployed {contract_type.value} contract at {contract_address}")
        return contract_address
    
    async def _deploy_solana_contract(self, contract_type: SmartContractType, 
                                    contract_bytecode: str, constructor_args: List[Any] = None) -> str:
        """Deploy Solana program"""
        # This is a simplified deployment - in production, you'd use proper Solana program deployment
        program_id = str(PublicKey.generate())
        
        self.logger.info(f"Deployed {contract_type.value} program at {program_id}")
        return program_id
    
    def get_account_balance(self) -> float:
        """Get account balance"""
        if not self.w3 or not self.account:
            return 0.0
        
        balance_wei = self.w3.eth.get_balance(self.account.address)
        balance_eth = self.w3.from_wei(balance_wei, 'ether')
        return float(balance_eth)


class SmartContractManager:
    """Smart contract manager for TruthGPT"""
    
    def __init__(self, blockchain_connector: BlockchainConnector):
        self.connector = blockchain_connector
        self.logger = logging.getLogger(f"SmartContractManager_{id(self)}")
        
        # Contract instances
        self.contracts: Dict[SmartContractType, str] = {}
        
        # Initialize default contracts
        self._init_default_contracts()
    
    def _init_default_contracts(self):
        """Initialize default smart contracts"""
        # In production, these would be actual deployed contract addresses
        self.contracts = {
            SmartContractType.MODEL_REGISTRY: "0x1234567890123456789012345678901234567890",
            SmartContractType.FEDERATED_LEARNING: "0x2345678901234567890123456789012345678901",
            SmartContractType.MODEL_MARKETPLACE: "0x3456789012345678901234567890123456789012",
            SmartContractType.GOVERNANCE: "0x4567890123456789012345678901234567890123",
            SmartContractType.STAKING: "0x5678901234567890123456789012345678901234"
        }
    
    async def register_model(self, model_metadata: ModelMetadata, model_data: bytes) -> str:
        """Register model on blockchain"""
        try:
            # Upload model to IPFS
            ipfs_hash = await self.connector.upload_to_ipfs(model_data)
            
            # Calculate model hash
            model_hash = self.connector.calculate_model_hash(model_data)
            
            # Update metadata with IPFS hash
            model_metadata.ipfs_hash = ipfs_hash
            model_metadata.model_hash = model_hash
            
            # Register on blockchain (simplified)
            transaction_id = await self._register_model_on_blockchain(model_metadata)
            
            self.logger.info(f"Registered model {model_metadata.model_id} with transaction {transaction_id}")
            return transaction_id
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    async def _register_model_on_blockchain(self, metadata: ModelMetadata) -> str:
        """Register model metadata on blockchain"""
        # This is a simplified implementation
        # In production, you'd interact with actual smart contracts
        
        transaction_data = {
            "model_id": metadata.model_id,
            "version_id": metadata.version_id,
            "model_hash": metadata.model_hash,
            "ipfs_hash": metadata.ipfs_hash,
            "creator": metadata.creator,
            "created_at": metadata.created_at,
            "price": metadata.price,
            "is_public": metadata.is_public
        }
        
        # Simulate transaction
        transaction_id = f"0x{hashlib.sha256(json.dumps(transaction_data, sort_keys=True).encode()).hexdigest()}"
        
        return transaction_id
    
    async def get_model_metadata(self, model_id: str, version_id: str) -> Optional[ModelMetadata]:
        """Get model metadata from blockchain"""
        # This is a simplified implementation
        # In production, you'd query actual smart contracts
        
        # Simulate blockchain query
        metadata_data = {
            "model_id": model_id,
            "version_id": version_id,
            "model_hash": f"hash_{model_id}_{version_id}",
            "ipfs_hash": f"ipfs_{model_id}_{version_id}",
            "creator": "0x1234567890123456789012345678901234567890",
            "created_at": int(time.time()),
            "model_type": "transformer",
            "price": 100,
            "is_public": True
        }
        
        return ModelMetadata(**metadata_data)
    
    async def download_model(self, model_id: str, version_id: str) -> bytes:
        """Download model from IPFS"""
        metadata = await self.get_model_metadata(model_id, version_id)
        if not metadata:
            raise Exception(f"Model {model_id}:{version_id} not found")
        
        return await self.connector.download_from_ipfs(metadata.ipfs_hash)
    
    async def create_federated_learning_session(self, session_id: str, 
                                              model_id: str, participants: List[str]) -> str:
        """Create federated learning session"""
        # This is a simplified implementation
        session_data = {
            "session_id": session_id,
            "model_id": model_id,
            "participants": participants,
            "created_at": int(time.time()),
            "status": "active"
        }
        
        transaction_id = f"0x{hashlib.sha256(json.dumps(session_data, sort_keys=True).encode()).hexdigest()}"
        
        self.logger.info(f"Created federated learning session {session_id}")
        return transaction_id
    
    async def submit_federated_update(self, session_id: str, participant: str, 
                                    model_update: bytes) -> str:
        """Submit federated learning update"""
        # Upload update to IPFS
        update_hash = await self.connector.upload_to_ipfs(model_update)
        
        # Submit to blockchain
        update_data = {
            "session_id": session_id,
            "participant": participant,
            "update_hash": update_hash,
            "timestamp": int(time.time())
        }
        
        transaction_id = f"0x{hashlib.sha256(json.dumps(update_data, sort_keys=True).encode()).hexdigest()}"
        
        self.logger.info(f"Submitted federated update for session {session_id}")
        return transaction_id


class ModelRegistryContract:
    """Model registry smart contract interface"""
    
    def __init__(self, smart_contract_manager: SmartContractManager):
        self.manager = smart_contract_manager
        self.logger = logging.getLogger(f"ModelRegistryContract_{id(self)}")
    
    async def register_model(self, model_metadata: ModelMetadata, model_data: bytes) -> str:
        """Register model in registry contract"""
        return await self.manager.register_model(model_metadata, model_data)
    
    async def get_model_info(self, model_id: str, version_id: str) -> Optional[ModelMetadata]:
        """Get model information from registry"""
        return await self.manager.get_model_metadata(model_id, version_id)
    
    async def list_models(self, creator: Optional[str] = None, 
                         model_type: Optional[str] = None) -> List[ModelMetadata]:
        """List models in registry"""
        # This is a simplified implementation
        models = []
        
        # Simulate querying registry
        for i in range(5):  # Simulate 5 models
            model_data = {
                "model_id": f"model_{i}",
                "version_id": f"v1.0.{i}",
                "model_hash": f"hash_{i}",
                "ipfs_hash": f"ipfs_{i}",
                "creator": f"0x{i:040x}",
                "created_at": int(time.time()) - i * 86400,
                "model_type": "transformer",
                "price": 100 + i * 50,
                "is_public": True
            }
            
            if creator and model_data["creator"] != creator:
                continue
            if model_type and model_data["model_type"] != model_type:
                continue
            
            models.append(ModelMetadata(**model_data))
        
        return models
    
    async def update_model_price(self, model_id: str, version_id: str, new_price: int) -> str:
        """Update model price"""
        # This is a simplified implementation
        transaction_id = f"0x{hashlib.sha256(f'update_price_{model_id}_{version_id}_{new_price}'.encode()).hexdigest()}"
        
        self.logger.info(f"Updated price for model {model_id}:{version_id} to {new_price}")
        return transaction_id
    
    async def transfer_model_ownership(self, model_id: str, version_id: str, 
                                    new_owner: str) -> str:
        """Transfer model ownership"""
        # This is a simplified implementation
        transaction_id = f"0x{hashlib.sha256(f'transfer_{model_id}_{version_id}_{new_owner}'.encode()).hexdigest()}"
        
        self.logger.info(f"Transferred ownership of model {model_id}:{version_id} to {new_owner}")
        return transaction_id


class IPFSManager:
    """IPFS manager for TruthGPT"""
    
    def __init__(self, blockchain_connector: BlockchainConnector):
        self.connector = blockchain_connector
        self.logger = logging.getLogger(f"IPFSManager_{id(self)}")
        
        # IPFS pinning service
        self.pinning_service = None
        self._init_pinning_service()
    
    def _init_pinning_service(self):
        """Initialize IPFS pinning service"""
        # In production, you'd use services like Pinata, Infura IPFS, etc.
        self.logger.info("IPFS pinning service initialized")
    
    async def pin_content(self, ipfs_hash: str) -> bool:
        """Pin content to IPFS"""
        try:
            # Simulate pinning
            self.logger.info(f"Pinned content {ipfs_hash}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pin content {ipfs_hash}: {e}")
            return False
    
    async def unpin_content(self, ipfs_hash: str) -> bool:
        """Unpin content from IPFS"""
        try:
            # Simulate unpinning
            self.logger.info(f"Unpinned content {ipfs_hash}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to unpin content {ipfs_hash}: {e}")
            return False
    
    async def get_content_info(self, ipfs_hash: str) -> Dict[str, Any]:
        """Get content information from IPFS"""
        try:
            # Simulate getting content info
            return {
                "hash": ipfs_hash,
                "size": 1024 * 1024,  # 1MB
                "pinned": True,
                "created_at": int(time.time())
            }
        except Exception as e:
            self.logger.error(f"Failed to get content info for {ipfs_hash}: {e}")
            return {}


class FederatedLearningContract:
    """Federated learning smart contract interface"""
    
    def __init__(self, smart_contract_manager: SmartContractManager):
        self.manager = smart_contract_manager
        self.logger = logging.getLogger(f"FederatedLearningContract_{id(self)}")
    
    async def create_session(self, session_id: str, model_id: str, 
                           participants: List[str], config: Dict[str, Any]) -> str:
        """Create federated learning session"""
        return await self.manager.create_federated_learning_session(session_id, model_id, participants)
    
    async def join_session(self, session_id: str, participant: str) -> bool:
        """Join federated learning session"""
        # This is a simplified implementation
        self.logger.info(f"Participant {participant} joined session {session_id}")
        return True
    
    async def submit_update(self, session_id: str, participant: str, 
                          model_update: bytes, metrics: Dict[str, float]) -> str:
        """Submit federated learning update"""
        return await self.manager.submit_federated_update(session_id, participant, model_update)
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get federated learning session status"""
        # This is a simplified implementation
        return {
            "session_id": session_id,
            "status": "active",
            "participants": 5,
            "updates_received": 3,
            "consensus_reached": False,
            "created_at": int(time.time()) - 3600,
            "last_update": int(time.time()) - 300
        }
    
    async def aggregate_updates(self, session_id: str) -> bytes:
        """Aggregate federated learning updates"""
        # This is a simplified implementation
        # In production, you'd implement proper federated averaging
        
        aggregated_model = b"aggregated_model_data"
        
        self.logger.info(f"Aggregated updates for session {session_id}")
        return aggregated_model


class TruthGPTBlockchainManager:
    """Unified blockchain manager for TruthGPT"""
    
    def __init__(self, config: BlockchainConfig):
        self.config = config
        self.logger = logging.getLogger(f"TruthGPTBlockchainManager_{id(self)}")
        
        # Initialize components
        self.connector = BlockchainConnector(config)
        self.smart_contract_manager = SmartContractManager(self.connector)
        self.model_registry = ModelRegistryContract(self.smart_contract_manager)
        self.ipfs_manager = IPFSManager(self.connector)
        self.federated_learning = FederatedLearningContract(self.smart_contract_manager)
        
        # Integration with TruthGPT components
        self.model_registry_local: Optional[Any] = None
        self.ai_enhancement: Optional[TruthGPTAIEnhancementManager] = None
    
    def set_model_registry(self, registry: Any):
        """Set local model registry for integration"""
        self.model_registry_local = registry
    
    def set_ai_enhancement(self, ai_enhancement: TruthGPTAIEnhancementManager):
        """Set AI enhancement manager for integration"""
        self.ai_enhancement = ai_enhancement
    
    async def deploy_model_to_blockchain(self, model_id: str, version_id: str, 
                                       model: TruthGPTModel, metadata: Dict[str, Any] = None) -> str:
        """Deploy model to blockchain"""
        try:
            # Serialize model
            model_data = pickle.dumps(model.state_dict())
            
            # Create model metadata
            model_metadata = ModelMetadata(
                model_id=model_id,
                version_id=version_id,
                model_hash=self.connector.calculate_model_hash(model_data),
                ipfs_hash="",  # Will be set during upload
                creator=self.connector.account.address if self.connector.account else "0x0",
                created_at=int(time.time()),
                model_type="truthgpt_transformer",
                parameters=metadata or {},
                is_public=True
            )
            
            # Register model on blockchain
            transaction_id = await self.model_registry.register_model(model_metadata, model_data)
            
            self.logger.info(f"Deployed model {model_id}:{version_id} to blockchain")
            return transaction_id
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model to blockchain: {e}")
            raise
    
    async def download_model_from_blockchain(self, model_id: str, version_id: str) -> TruthGPTModel:
        """Download model from blockchain"""
        try:
            # Get model metadata
            metadata = await self.model_registry.get_model_info(model_id, version_id)
            if not metadata:
                raise Exception(f"Model {model_id}:{version_id} not found on blockchain")
            
            # Download model data
            model_data = await self.smart_contract_manager.download_model(model_id, version_id)
            
            # Deserialize model
            model_state_dict = pickle.loads(model_data)
            
            # Create model instance
            model_config = TruthGPTModelConfig(
                vocab_size=metadata.parameters.get("vocab_size", 1000),
                hidden_size=metadata.parameters.get("hidden_size", 256),
                num_layers=metadata.parameters.get("num_layers", 2),
                num_heads=metadata.parameters.get("num_heads", 4)
            )
            
            model = TruthGPTModel(model_config)
            model.load_state_dict(model_state_dict)
            
            self.logger.info(f"Downloaded model {model_id}:{version_id} from blockchain")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to download model from blockchain: {e}")
            raise
    
    async def start_federated_learning(self, model_id: str, participants: List[str], 
                                     config: Dict[str, Any] = None) -> str:
        """Start federated learning session"""
        session_id = str(uuid.uuid4())
        
        # Create federated learning session
        transaction_id = await self.federated_learning.create_session(
            session_id, model_id, participants, config or {}
        )
        
        self.logger.info(f"Started federated learning session {session_id}")
        return session_id
    
    async def participate_in_federated_learning(self, session_id: str, 
                                              model_update: bytes, 
                                              metrics: Dict[str, float]) -> str:
        """Participate in federated learning"""
        participant_id = self.connector.account.address if self.connector.account else "anonymous"
        
        # Submit update
        transaction_id = await self.federated_learning.submit_update(
            session_id, participant_id, model_update, metrics
        )
        
        self.logger.info(f"Submitted federated learning update for session {session_id}")
        return transaction_id
    
    async def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        return {
            "blockchain_type": self.config.blockchain_type.value,
            "account_balance": self.connector.get_account_balance(),
            "connected": self.connector.w3 is not None,
            "ipfs_connected": self.connector.ipfs_client is not None,
            "contracts_deployed": len(self.smart_contract_manager.contracts),
            "federated_sessions": 0,  # Would track actual sessions
            "models_on_blockchain": 0  # Would track actual models
        }
    
    async def sync_with_local_registry(self):
        """Sync blockchain models with local registry"""
        if not self.model_registry_local:
            self.logger.warning("No local registry set for sync")
            return
        
        try:
            # Get all models from blockchain
            blockchain_models = await self.model_registry.list_models()
            
            # Sync with local registry
            for model_metadata in blockchain_models:
                # Check if model exists in local registry
                local_versions = self.model_registry_local.get_model_versions(model_metadata.model_id)
                
                if not local_versions:
                    # Download and add to local registry
                    model = await self.download_model_from_blockchain(
                        model_metadata.model_id, model_metadata.version_id
                    )
                    
                    # Add to local registry
                    self.model_registry_local.create_version(
                        model_metadata.model_id, model, 
                        f"Synced from blockchain - {model_metadata.version_id}"
                    )
            
            self.logger.info(f"Synced {len(blockchain_models)} models from blockchain")
            
        except Exception as e:
            self.logger.error(f"Failed to sync with local registry: {e}")


def create_blockchain_manager(
    config: Optional[BlockchainConfig] = None
) -> TruthGPTBlockchainManager:
    """Create blockchain manager with default configuration"""
    if config is None:
        config = BlockchainConfig()
    
    return TruthGPTBlockchainManager(config)


def create_blockchain_connector(
    config: Optional[BlockchainConfig] = None
) -> BlockchainConnector:
    """Create blockchain connector"""
    if config is None:
        config = BlockchainConfig()
    
    return BlockchainConnector(config)


def create_ipfs_manager(
    blockchain_connector: BlockchainConnector
) -> IPFSManager:
    """Create IPFS manager"""
    return IPFSManager(blockchain_connector)


def create_model_registry_contract(
    smart_contract_manager: SmartContractManager
) -> ModelRegistryContract:
    """Create model registry contract"""
    return ModelRegistryContract(smart_contract_manager)


def create_federated_learning_contract(
    smart_contract_manager: SmartContractManager
) -> FederatedLearningContract:
    """Create federated learning contract"""
    return FederatedLearningContract(smart_contract_manager)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create blockchain manager
        blockchain_config = BlockchainConfig(
            blockchain_type=BlockchainType.ETHEREUM,
            enable_ipfs=True,
            enable_encryption=True
        )
        
        blockchain_manager = create_blockchain_manager(blockchain_config)
        
        # Example: Deploy model to blockchain
        model_config = TruthGPTModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4
        )
        model = TruthGPTModel(model_config)
        
        transaction_id = await blockchain_manager.deploy_model_to_blockchain(
            "test_model", "v1.0.0", model, {"description": "Test model"}
        )
        
        print(f"Model deployed with transaction: {transaction_id}")
        
        # Get blockchain stats
        stats = await blockchain_manager.get_blockchain_stats()
        print(f"Blockchain stats: {stats}")
    
    # Run example
    asyncio.run(main())

