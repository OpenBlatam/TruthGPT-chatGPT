"""
TruthGPT Federated Learning and Decentralized AI Networks
Advanced federated learning, decentralized AI networks, and collaborative intelligence for TruthGPT
"""

import asyncio
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import pickle
import threading
from datetime import datetime, timedelta
import uuid
import math
import random
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import heapq
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import hashlib
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Import TruthGPT modules
from .models import TruthGPTModel, TruthGPTModelConfig
from .orchestration import AIOrchestrator, AIAgent, AgentType
from .blockchain import TruthGPTBlockchainManager, BlockchainConfig
from .ai_enhancement import TruthGPTAIEnhancementManager


class FederationType(Enum):
    """Types of federated learning"""
    HORIZONTAL_FEDERATION = "horizontal"  # Same features, different samples
    VERTICAL_FEDERATION = "vertical"      # Different features, same samples
    FEDERATED_TRANSFER_LEARNING = "transfer"  # Transfer learning across domains
    FEDERATED_META_LEARNING = "meta"      # Meta-learning across federations
    HIERARCHICAL_FEDERATION = "hierarchical"  # Multi-level federation
    CROSS_SILO_FEDERATION = "cross_silo"  # Cross-organizational federation


class AggregationMethod(Enum):
    """Federated aggregation methods"""
    FEDERATED_AVERAGING = "fedavg"
    FEDERATED_SGD = "fedsgd"
    FEDERATED_PROX = "fedprox"
    SCAFFOLD = "scaffold"
    ADAPTIVE_FEDERATED_AVERAGING = "adaptive_fedavg"
    PERSONALIZED_FEDERATED_LEARNING = "personalized"
    DIFFERENTIAL_PRIVATE_FEDERATED_AVERAGING = "dp_fedavg"


class NetworkTopology(Enum):
    """Network topologies for decentralized learning"""
    STAR = "star"                    # Central server
    RING = "ring"                   # Ring topology
    MESH = "mesh"                   # Fully connected
    TREE = "tree"                   # Tree topology
    GRID = "grid"                   # Grid topology
    SMALL_WORLD = "small_world"     # Small-world network
    SCALE_FREE = "scale_free"       # Scale-free network
    HIERARCHICAL = "hierarchical"   # Hierarchical network


class NodeRole(Enum):
    """Roles in federated network"""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    OBSERVER = "observer"
    GATEWAY = "gateway"


class PrivacyLevel(Enum):
    """Privacy levels for federated learning"""
    NO_PRIVACY = "no_privacy"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    MULTI_PARTY_COMPUTATION = "mpc"
    FEDERATED_LEARNING_WITH_SECURE_AGGREGATION = "fl_sa"


@dataclass
class FederationConfig:
    """Configuration for federated learning"""
    federation_type: FederationType = FederationType.HORIZONTAL_FEDERATION
    aggregation_method: AggregationMethod = AggregationMethod.FEDERATED_AVERAGING
    network_topology: NetworkTopology = NetworkTopology.STAR
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    num_rounds: int = 100
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    participation_rate: float = 0.8
    min_participants: int = 3
    max_participants: int = 100
    communication_rounds: int = 10
    enable_secure_aggregation: bool = True
    enable_differential_privacy: bool = True
    noise_multiplier: float = 1.0
    l2_norm_clip: float = 1.0
    enable_byzantine_robustness: bool = True
    byzantine_threshold: float = 0.3


@dataclass
class NodeConfig:
    """Configuration for federated node"""
    node_id: str
    node_role: NodeRole = NodeRole.PARTICIPANT
    capabilities: List[str] = field(default_factory=list)
    data_size: int = 1000
    compute_power: float = 1.0
    bandwidth: float = 1.0
    reliability: float = 0.95
    privacy_budget: float = 1.0
    enable_local_training: bool = True
    enable_model_sharing: bool = True
    enable_gradient_sharing: bool = True
    local_model_path: Optional[str] = None
    encryption_key: Optional[str] = None


@dataclass
class FederationRound:
    """Federated learning round"""
    round_id: int
    participants: List[str]
    global_model_version: str
    local_updates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aggregated_model: Optional[Dict[str, Any]] = None
    round_metrics: Dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: bool = False


@dataclass
class ModelUpdate:
    """Model update for federated learning"""
    node_id: str
    model_weights: Dict[str, torch.Tensor]
    gradients: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    privacy_cost: float = 0.0
    validation_score: float = 0.0


class SecureAggregator:
    """Secure aggregation for federated learning"""
    
    def __init__(self, config: FederationConfig):
        self.config = config
        self.logger = logging.getLogger(f"SecureAggregator_{id(self)}")
        
        # Encryption setup
        self.encryption_keys: Dict[str, bytes] = {}
        self.shared_secrets: Dict[str, bytes] = {}
        
        # Secure aggregation state
        self.aggregation_state: Dict[str, Any] = {}
        self.participant_keys: Dict[str, str] = {}
    
    def setup_encryption(self, participants: List[str]) -> Dict[str, str]:
        """Setup encryption for secure aggregation"""
        # Generate encryption keys for each participant
        for participant in participants:
            key = Fernet.generate_key()
            self.encryption_keys[participant] = key
            self.participant_keys[participant] = base64.urlsafe_b64encode(key).decode()
        
        # Generate shared secrets for pairwise communication
        for i, participant1 in enumerate(participants):
            for participant2 in participants[i+1:]:
                shared_secret = self._generate_shared_secret(participant1, participant2)
                self.shared_secrets[f"{participant1}_{participant2}"] = shared_secret
        
        return self.participant_keys
    
    def _generate_shared_secret(self, participant1: str, participant2: str) -> bytes:
        """Generate shared secret between two participants"""
        # Simplified shared secret generation
        combined_id = f"{participant1}_{participant2}"
        secret = hashlib.sha256(combined_id.encode()).digest()
        return secret
    
    def encrypt_model_update(self, node_id: str, model_update: ModelUpdate) -> bytes:
        """Encrypt model update"""
        if node_id not in self.encryption_keys:
            raise Exception(f"No encryption key for node {node_id}")
        
        # Serialize model update
        update_data = {
            "model_weights": {k: v.tolist() for k, v in model_update.model_weights.items()},
            "gradients": {k: v.tolist() for k, v in model_update.gradients.items()},
            "metadata": model_update.metadata,
            "timestamp": model_update.timestamp
        }
        
        # Encrypt data
        fernet = Fernet(self.encryption_keys[node_id])
        encrypted_data = fernet.encrypt(json.dumps(update_data).encode())
        
        return encrypted_data
    
    def decrypt_model_update(self, node_id: str, encrypted_data: bytes) -> ModelUpdate:
        """Decrypt model update"""
        if node_id not in self.encryption_keys:
            raise Exception(f"No encryption key for node {node_id}")
        
        # Decrypt data
        fernet = Fernet(self.encryption_keys[node_id])
        decrypted_data = fernet.decrypt(encrypted_data)
        update_data = json.loads(decrypted_data.decode())
        
        # Reconstruct model update
        model_update = ModelUpdate(
            node_id=node_id,
            model_weights={k: torch.tensor(v) for k, v in update_data["model_weights"].items()},
            gradients={k: torch.tensor(v) for k, v in update_data["gradients"].items()},
            metadata=update_data["metadata"],
            timestamp=update_data["timestamp"]
        )
        
        return model_update
    
    def secure_aggregate(self, encrypted_updates: Dict[str, bytes]) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation"""
        # Decrypt all updates
        decrypted_updates = {}
        for node_id, encrypted_data in encrypted_updates.items():
            decrypted_updates[node_id] = self.decrypt_model_update(node_id, encrypted_data)
        
        # Perform aggregation
        aggregated_weights = self._aggregate_weights(decrypted_updates)
        
        return aggregated_weights
    
    def _aggregate_weights(self, updates: Dict[str, ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate model weights"""
        if not updates:
            return {}
        
        # Get first update to determine structure
        first_update = next(iter(updates.values()))
        aggregated_weights = {}
        
        # Initialize aggregated weights
        for param_name in first_update.model_weights.keys():
            aggregated_weights[param_name] = torch.zeros_like(first_update.model_weights[param_name])
        
        # Sum all updates
        total_weight = 0.0
        for update in updates.values():
            weight = update.metadata.get("weight", 1.0)
            total_weight += weight
            
            for param_name, param_tensor in update.model_weights.items():
                aggregated_weights[param_name] += weight * param_tensor
        
        # Average the weights
        if total_weight > 0:
            for param_name in aggregated_weights:
                aggregated_weights[param_name] /= total_weight
        
        return aggregated_weights


class DifferentialPrivacyEngine:
    """Differential privacy engine for federated learning"""
    
    def __init__(self, config: FederationConfig):
        self.config = config
        self.logger = logging.getLogger(f"DifferentialPrivacyEngine_{id(self)}")
        
        # Privacy parameters
        self.epsilon = config.noise_multiplier
        self.delta = 1e-5
        self.sensitivity = config.l2_norm_clip
        
        # Privacy accounting
        self.privacy_budget: Dict[str, float] = {}
        self.privacy_history: List[Dict[str, Any]] = []
    
    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor], 
                             node_id: str) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients"""
        noisy_gradients = {}
        
        for param_name, gradient in gradients.items():
            # Clip gradients
            clipped_gradient = self._clip_gradient(gradient)
            
            # Add Gaussian noise
            noise = torch.normal(0, self.sensitivity * self.epsilon, size=gradient.shape)
            noisy_gradients[param_name] = clipped_gradient + noise
        
        # Update privacy budget
        self._update_privacy_budget(node_id, self.epsilon)
        
        return noisy_gradients
    
    def _clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Clip gradient to L2 norm"""
        gradient_norm = torch.norm(gradient)
        if gradient_norm > self.sensitivity:
            gradient = gradient * (self.sensitivity / gradient_norm)
        return gradient
    
    def _update_privacy_budget(self, node_id: str, epsilon_cost: float):
        """Update privacy budget for node"""
        if node_id not in self.privacy_budget:
            self.privacy_budget[node_id] = 1.0  # Initial budget
        
        self.privacy_budget[node_id] -= epsilon_cost
        
        # Record privacy usage
        self.privacy_history.append({
            "node_id": node_id,
            "epsilon_cost": epsilon_cost,
            "remaining_budget": self.privacy_budget[node_id],
            "timestamp": time.time()
        })
    
    def check_privacy_budget(self, node_id: str) -> bool:
        """Check if node has sufficient privacy budget"""
        return self.privacy_budget.get(node_id, 0) > self.epsilon
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get privacy statistics"""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "privacy_budgets": self.privacy_budget,
            "total_privacy_usage": len(self.privacy_history)
        }


class FederatedNode:
    """Federated learning node"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = logging.getLogger(f"FederatedNode_{config.node_id}")
        
        # Node state
        self.local_model: Optional[TruthGPTModel] = None
        self.local_data: List[Dict[str, Any]] = []
        self.training_history: List[Dict[str, Any]] = []
        
        # Federation components
        self.secure_aggregator: Optional[SecureAggregator] = None
        self.privacy_engine: Optional[DifferentialPrivacyEngine] = None
        
        # Communication
        self.message_queue: deque = deque()
        self.neighbors: Set[str] = set()
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {}
        self.participation_history: List[Dict[str, Any]] = []
    
    def initialize_model(self, model_config: TruthGPTModelConfig):
        """Initialize local model"""
        self.local_model = TruthGPTModel(model_config)
        self.logger.info(f"Initialized local model for node {self.config.node_id}")
    
    def load_local_data(self, data: List[Dict[str, Any]]):
        """Load local training data"""
        self.local_data = data
        self.logger.info(f"Loaded {len(data)} samples for node {self.config.node_id}")
    
    async def local_training(self, epochs: int = 5) -> ModelUpdate:
        """Perform local training"""
        if not self.local_model:
            raise Exception("Local model not initialized")
        
        self.logger.info(f"Starting local training for {epochs} epochs")
        
        # Simulate local training
        training_losses = []
        for epoch in range(epochs):
            # Simulate training step
            loss = random.uniform(0.1, 1.0) * (1 - epoch / epochs)
            training_losses.append(loss)
        
        # Create model update
        model_update = ModelUpdate(
            node_id=self.config.node_id,
            model_weights=self._extract_model_weights(),
            gradients=self._extract_gradients(),
            metadata={
                "epochs": epochs,
                "final_loss": training_losses[-1],
                "data_size": len(self.local_data),
                "weight": len(self.local_data)  # Weight based on data size
            }
        )
        
        # Record training history
        self.training_history.append({
            "epochs": epochs,
            "final_loss": training_losses[-1],
            "timestamp": time.time()
        })
        
        return model_update
    
    def _extract_model_weights(self) -> Dict[str, torch.Tensor]:
        """Extract model weights"""
        if not self.local_model:
            return {}
        
        weights = {}
        for name, param in self.local_model.named_parameters():
            weights[name] = param.data.clone()
        
        return weights
    
    def _extract_gradients(self) -> Dict[str, torch.Tensor]:
        """Extract gradients"""
        if not self.local_model:
            return {}
        
        gradients = {}
        for name, param in self.local_model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
            else:
                gradients[name] = torch.zeros_like(param.data)
        
        return gradients
    
    def update_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Update local model with aggregated weights"""
        if not self.local_model:
            raise Exception("Local model not initialized")
        
        # Update model parameters
        for name, param in self.local_model.named_parameters():
            if name in aggregated_weights:
                param.data = aggregated_weights[name].clone()
        
        self.logger.info(f"Updated local model for node {self.config.node_id}")
    
    async def participate_in_round(self, round_id: int, 
                                federation_config: FederationConfig) -> ModelUpdate:
        """Participate in federated learning round"""
        self.logger.info(f"Participating in federation round {round_id}")
        
        # Perform local training
        model_update = await self.local_training(federation_config.local_epochs)
        
        # Apply differential privacy if enabled
        if federation_config.enable_differential_privacy and self.privacy_engine:
            if self.privacy_engine.check_privacy_budget(self.config.node_id):
                model_update.gradients = self.privacy_engine.add_noise_to_gradients(
                    model_update.gradients, self.config.node_id
                )
            else:
                self.logger.warning(f"Insufficient privacy budget for node {self.config.node_id}")
                return None
        
        # Record participation
        self.participation_history.append({
            "round_id": round_id,
            "timestamp": time.time(),
            "data_size": len(self.local_data),
            "privacy_cost": model_update.privacy_cost
        })
        
        return model_update
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get node statistics"""
        return {
            "node_id": self.config.node_id,
            "node_role": self.config.node_role.value,
            "data_size": len(self.local_data),
            "training_rounds": len(self.training_history),
            "participation_rounds": len(self.participation_history),
            "neighbors": len(self.neighbors),
            "performance_metrics": self.performance_metrics
        }


class DecentralizedAINetwork:
    """Decentralized AI network for TruthGPT"""
    
    def __init__(self, config: FederationConfig):
        self.config = config
        self.logger = logging.getLogger(f"DecentralizedAINetwork_{id(self)}")
        
        # Network components
        self.nodes: Dict[str, FederatedNode] = {}
        self.network_graph: nx.Graph = nx.Graph()
        
        # Federation management
        self.federation_rounds: List[FederationRound] = []
        self.global_model: Optional[TruthGPTModel] = None
        
        # Security and privacy
        self.secure_aggregator = SecureAggregator(config)
        self.privacy_engine = DifferentialPrivacyEngine(config)
        
        # Network topology
        self._init_network_topology()
        
        # Performance tracking
        self.network_metrics: Dict[str, Any] = {
            "total_rounds": 0,
            "successful_rounds": 0,
            "average_participation": 0.0,
            "model_convergence": 0.0,
            "privacy_cost": 0.0
        }
    
    def _init_network_topology(self):
        """Initialize network topology"""
        if self.config.network_topology == NetworkTopology.STAR:
            self._create_star_topology()
        elif self.config.network_topology == NetworkTopology.RING:
            self._create_ring_topology()
        elif self.config.network_topology == NetworkTopology.MESH:
            self._create_mesh_topology()
        elif self.config.network_topology == NetworkTopology.TREE:
            self._create_tree_topology()
        else:
            self._create_default_topology()
    
    def _create_star_topology(self):
        """Create star topology with central coordinator"""
        # This will be populated when nodes are added
        pass
    
    def _create_ring_topology(self):
        """Create ring topology"""
        # This will be populated when nodes are added
        pass
    
    def _create_mesh_topology(self):
        """Create fully connected mesh topology"""
        # This will be populated when nodes are added
        pass
    
    def _create_tree_topology(self):
        """Create tree topology"""
        # This will be populated when nodes are added
        pass
    
    def _create_default_topology(self):
        """Create default topology"""
        pass
    
    def add_node(self, config: NodeConfig) -> str:
        """Add node to network"""
        node = FederatedNode(config)
        self.nodes[config.node_id] = node
        
        # Add to network graph
        self.network_graph.add_node(config.node_id, **config.__dict__)
        
        # Update topology based on network type
        self._update_topology(config.node_id)
        
        self.logger.info(f"Added node {config.node_id} to network")
        return config.node_id
    
    def _update_topology(self, new_node_id: str):
        """Update network topology when new node is added"""
        if self.config.network_topology == NetworkTopology.STAR:
            # Connect all nodes to coordinator
            coordinator_nodes = [n for n, data in self.network_graph.nodes(data=True) 
                               if data.get('node_role') == NodeRole.COORDINATOR.value]
            if coordinator_nodes:
                coordinator = coordinator_nodes[0]
                self.network_graph.add_edge(new_node_id, coordinator)
        
        elif self.config.network_topology == NetworkTopology.RING:
            # Connect in ring pattern
            nodes = list(self.network_graph.nodes())
            if len(nodes) > 1:
                prev_node = nodes[-2]
                self.network_graph.add_edge(prev_node, new_node_id)
                if len(nodes) > 2:
                    next_node = nodes[0]
                    self.network_graph.add_edge(new_node_id, next_node)
        
        elif self.config.network_topology == NetworkTopology.MESH:
            # Connect to all existing nodes
            for existing_node in self.network_graph.nodes():
                if existing_node != new_node_id:
                    self.network_graph.add_edge(existing_node, new_node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node from network"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.network_graph.remove_node(node_id)
            self.logger.info(f"Removed node {node_id} from network")
            return True
        return False
    
    async def run_federation_round(self, round_id: int) -> FederationRound:
        """Run a federated learning round"""
        self.logger.info(f"Starting federation round {round_id}")
        
        # Create federation round
        federation_round = FederationRound(
            round_id=round_id,
            participants=list(self.nodes.keys()),
            global_model_version=f"v{round_id}"
        )
        
        # Select participants
        participants = self._select_participants()
        federation_round.participants = participants
        
        if len(participants) < self.config.min_participants:
            self.logger.warning(f"Insufficient participants: {len(participants)}")
            federation_round.success = False
            return federation_round
        
        # Collect local updates
        local_updates = {}
        encrypted_updates = {}
        
        for participant_id in participants:
            try:
                # Get local update
                node = self.nodes[participant_id]
                model_update = await node.participate_in_round(round_id, self.config)
                
                if model_update:
                    local_updates[participant_id] = model_update
                    
                    # Encrypt update if secure aggregation is enabled
                    if self.config.enable_secure_aggregation:
                        encrypted_data = self.secure_aggregator.encrypt_model_update(
                            participant_id, model_update
                        )
                        encrypted_updates[participant_id] = encrypted_data
                
            except Exception as e:
                self.logger.error(f"Node {participant_id} failed to participate: {e}")
        
        # Aggregate updates
        if self.config.enable_secure_aggregation and encrypted_updates:
            aggregated_weights = self.secure_aggregator.secure_aggregate(encrypted_updates)
        else:
            aggregated_weights = self._aggregate_updates(local_updates)
        
        # Update global model
        if aggregated_weights:
            self._update_global_model(aggregated_weights)
            
            # Distribute updated model to participants
            for participant_id in participants:
                if participant_id in self.nodes:
                    self.nodes[participant_id].update_model(aggregated_weights)
        
        # Calculate round metrics
        federation_round.aggregated_model = aggregated_weights
        federation_round.round_metrics = self._calculate_round_metrics(local_updates)
        federation_round.end_time = time.time()
        federation_round.success = True
        
        # Store federation round
        self.federation_rounds.append(federation_round)
        
        # Update network metrics
        self._update_network_metrics(federation_round)
        
        self.logger.info(f"Completed federation round {round_id}")
        return federation_round
    
    def _select_participants(self) -> List[str]:
        """Select participants for federation round"""
        all_nodes = list(self.nodes.keys())
        
        # Random selection based on participation rate
        num_participants = int(len(all_nodes) * self.config.participation_rate)
        num_participants = max(num_participants, self.config.min_participants)
        num_participants = min(num_participants, self.config.max_participants)
        
        participants = random.sample(all_nodes, num_participants)
        return participants
    
    def _aggregate_updates(self, updates: Dict[str, ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate model updates"""
        if not updates:
            return {}
        
        # Weighted averaging based on data size
        total_weight = sum(update.metadata.get("weight", 1.0) for update in updates.values())
        
        aggregated_weights = {}
        first_update = next(iter(updates.values()))
        
        # Initialize aggregated weights
        for param_name in first_update.model_weights.keys():
            aggregated_weights[param_name] = torch.zeros_like(first_update.model_weights[param_name])
        
        # Aggregate weights
        for update in updates.values():
            weight = update.metadata.get("weight", 1.0) / total_weight
            
            for param_name, param_tensor in update.model_weights.items():
                aggregated_weights[param_name] += weight * param_tensor
        
        return aggregated_weights
    
    def _update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Update global model"""
        if not self.global_model:
            # Initialize global model
            model_config = TruthGPTModelConfig(
                vocab_size=1000,
                hidden_size=256,
                num_layers=2,
                num_heads=4
            )
            self.global_model = TruthGPTModel(model_config)
        
        # Update global model parameters
        for name, param in self.global_model.named_parameters():
            if name in aggregated_weights:
                param.data = aggregated_weights[name].clone()
    
    def _calculate_round_metrics(self, updates: Dict[str, ModelUpdate]) -> Dict[str, float]:
        """Calculate round metrics"""
        if not updates:
            return {}
        
        metrics = {
            "participation_rate": len(updates) / len(self.nodes),
            "average_loss": np.mean([u.metadata.get("final_loss", 0) for u in updates.values()]),
            "total_data_size": sum([u.metadata.get("data_size", 0) for u in updates.values()]),
            "privacy_cost": sum([u.privacy_cost for u in updates.values()])
        }
        
        return metrics
    
    def _update_network_metrics(self, federation_round: FederationRound):
        """Update network metrics"""
        self.network_metrics["total_rounds"] += 1
        
        if federation_round.success:
            self.network_metrics["successful_rounds"] += 1
        
        # Update average participation
        total_participation = sum([
            round.round_metrics.get("participation_rate", 0) 
            for round in self.federation_rounds
        ])
        self.network_metrics["average_participation"] = total_participation / len(self.federation_rounds)
        
        # Update privacy cost
        self.network_metrics["privacy_cost"] = sum([
            round.round_metrics.get("privacy_cost", 0) 
            for round in self.federation_rounds
        ])
    
    async def run_federated_learning(self, num_rounds: int = None) -> Dict[str, Any]:
        """Run federated learning for specified number of rounds"""
        num_rounds = num_rounds or self.config.num_rounds
        
        self.logger.info(f"Starting federated learning for {num_rounds} rounds")
        
        start_time = time.time()
        successful_rounds = 0
        
        for round_id in range(num_rounds):
            try:
                federation_round = await self.run_federation_round(round_id)
                if federation_round.success:
                    successful_rounds += 1
                
                # Log progress
                if round_id % 10 == 0:
                    self.logger.info(f"Completed {round_id}/{num_rounds} rounds")
                
            except Exception as e:
                self.logger.error(f"Federation round {round_id} failed: {e}")
        
        execution_time = time.time() - start_time
        
        return {
            "total_rounds": num_rounds,
            "successful_rounds": successful_rounds,
            "success_rate": successful_rounds / num_rounds,
            "execution_time": execution_time,
            "network_metrics": self.network_metrics,
            "privacy_stats": self.privacy_engine.get_privacy_stats()
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        return {
            "total_nodes": len(self.nodes),
            "network_topology": self.config.network_topology.value,
            "federation_type": self.config.federation_type.value,
            "aggregation_method": self.config.aggregation_method.value,
            "privacy_level": self.config.privacy_level.value,
            "network_metrics": self.network_metrics,
            "graph_nodes": self.network_graph.number_of_nodes(),
            "graph_edges": self.network_graph.number_of_edges(),
            "node_stats": {node_id: node.get_node_stats() for node_id, node in self.nodes.items()}
        }


def create_decentralized_ai_network(config: FederationConfig) -> DecentralizedAINetwork:
    """Create decentralized AI network"""
    return DecentralizedAINetwork(config)


def create_federated_node(config: NodeConfig) -> FederatedNode:
    """Create federated node"""
    return FederatedNode(config)


def create_secure_aggregator(config: FederationConfig) -> SecureAggregator:
    """Create secure aggregator"""
    return SecureAggregator(config)


def create_differential_privacy_engine(config: FederationConfig) -> DifferentialPrivacyEngine:
    """Create differential privacy engine"""
    return DifferentialPrivacyEngine(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create federation config
        federation_config = FederationConfig(
            federation_type=FederationType.HORIZONTAL_FEDERATION,
            aggregation_method=AggregationMethod.FEDERATED_AVERAGING,
            network_topology=NetworkTopology.STAR,
            privacy_level=PrivacyLevel.DIFFERENTIAL_PRIVACY,
            num_rounds=10,
            local_epochs=3,
            enable_secure_aggregation=True,
            enable_differential_privacy=True
        )
        
        # Create decentralized network
        network = create_decentralized_ai_network(federation_config)
        
        # Add nodes
        for i in range(5):
            node_config = NodeConfig(
                node_id=f"node_{i}",
                node_role=NodeRole.PARTICIPANT,
                capabilities=["training", "inference"],
                data_size=random.randint(100, 1000)
            )
            network.add_node(node_config)
        
        # Run federated learning
        result = await network.run_federated_learning(num_rounds=5)
        print(f"Federated learning result: {result}")
        
        # Get network stats
        stats = network.get_network_stats()
        print(f"Network stats: {stats}")
    
    # Run example
    asyncio.run(main())

