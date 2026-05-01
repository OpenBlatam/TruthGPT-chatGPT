"""
Federated Learning Module
Advanced federated learning capabilities for TruthGPT optimization
"""

import torch
import torch.nn as nn
import numpy as np
import random
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import queue
from collections import defaultdict

logger = logging.getLogger(__name__)

class FederatedStrategy(Enum):
    """Federated learning strategies."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    SECURE_AGGREGATION = "secure_aggregation"
    FEDERATED_DROPOUT = "federated_dropout"
    FEDERATED_MATCHING = "federated_matching"

class AggregationMethod(Enum):
    """Aggregation methods for federated learning."""
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    KRUM = "krum"
    BYZANTINE_ROBUST = "byzantine_robust"
    DIFFERENTIAL_PRIVACY = "differential_privacy"

@dataclass
class ClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    data_size: int
    local_epochs: int = 5
    learning_rate: float = 1e-3
    batch_size: int = 32
    device: str = "cpu"
    enable_differential_privacy: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0

@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    strategy: FederatedStrategy = FederatedStrategy.FEDAVG
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    num_clients: int = 10
    num_rounds: int = 100
    participation_rate: float = 1.0
    communication_rounds: int = 10
    enable_secure_aggregation: bool = False
    enable_byzantine_robustness: bool = False
    byzantine_threshold: float = 0.1
    enable_differential_privacy: bool = False
    privacy_budget: float = 1.0
    enable_compression: bool = False
    compression_ratio: float = 0.1

class BaseFederatedManager(ABC):
    """Base class for federated learning managers."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.global_model: Optional[nn.Module] = None
        self.client_models: Dict[str, nn.Module] = {}
        self.client_configs: Dict[str, ClientConfig] = {}
        self.round_history: List[Dict[str, Any]] = []
        self.current_round = 0
    
    @abstractmethod
    def aggregate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates."""
        pass
    
    def initialize_clients(self, client_configs: List[ClientConfig]):
        """Initialize federated learning clients."""
        for client_config in client_configs:
            self.client_configs[client_config.client_id] = client_config
            self.logger.info(f"Initialized client: {client_config.client_id}")
    
    def distribute_model(self, global_model: nn.Module):
        """Distribute global model to clients."""
        self.global_model = global_model
        
        for client_id in self.client_configs.keys():
            # Create copy of global model for each client
            client_model = copy.deepcopy(global_model)
            self.client_models[client_id] = client_model
            self.logger.info(f"Distributed model to client: {client_id}")
    
    def collect_updates(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Collect updates from clients."""
        client_updates = {}
        
        for client_id, client_model in self.client_models.items():
            # Simulate client training
            client_updates[client_id] = self._simulate_client_update(client_id, client_model)
        
        return client_updates
    
    def _simulate_client_update(self, client_id: str, client_model: nn.Module) -> Dict[str, torch.Tensor]:
        """Simulate client model update."""
        client_config = self.client_configs[client_id]
        
        # Simulate local training
        updates = {}
        for name, param in client_model.named_parameters():
            # Add some noise to simulate training
            noise = torch.randn_like(param) * 0.01
            updates[name] = param.data + noise
        
        return updates
    
    def update_global_model(self, aggregated_updates: Dict[str, torch.Tensor]):
        """Update global model with aggregated updates."""
        if self.global_model is None:
            return
        
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_updates:
                    param.data = aggregated_updates[name]
    
    def run_federated_round(self) -> Dict[str, Any]:
        """Run one federated learning round."""
        self.logger.info(f"Starting federated round {self.current_round + 1}")
        
        # Collect updates from clients
        client_updates = self.collect_updates()
        
        # Aggregate updates
        aggregated_updates = self.aggregate_updates(client_updates)
        
        # Update global model
        self.update_global_model(aggregated_updates)
        
        # Distribute updated model to clients
        self.distribute_model(self.global_model)
        
        # Record round metrics
        round_metrics = {
            'round': self.current_round,
            'num_participants': len(client_updates),
            'timestamp': time.time()
        }
        
        self.round_history.append(round_metrics)
        self.current_round += 1
        
        return round_metrics
    
    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get federated learning round history."""
        return self.round_history.copy()

class FedAvgManager(BaseFederatedManager):
    """Federated Averaging (FedAvg) manager."""
    
    def __init__(self, config: FederatedConfig):
        super().__init__(config)
        self.strategy = FederatedStrategy.FEDAVG
    
    def aggregate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate updates using FedAvg."""
        self.logger.info("Aggregating updates using FedAvg")
        
        if not client_updates:
            return {}
        
        # Calculate weighted average
        aggregated_updates = {}
        total_weight = 0
        
        # Calculate total data size
        for client_id in client_updates.keys():
            client_config = self.client_configs[client_id]
            total_weight += client_config.data_size
        
        # Aggregate each parameter
        for param_name in next(iter(client_updates.values())).keys():
            weighted_sum = None
            
            for client_id, updates in client_updates.items():
                client_config = self.client_configs[client_id]
                weight = client_config.data_size / total_weight
                
                if weighted_sum is None:
                    weighted_sum = updates[param_name] * weight
                else:
                    weighted_sum += updates[param_name] * weight
            
            aggregated_updates[param_name] = weighted_sum
        
        return aggregated_updates

class FedProxManager(BaseFederatedManager):
    """FedProx manager with proximal term."""
    
    def __init__(self, config: FederatedConfig):
        super().__init__(config)
        self.strategy = FederatedStrategy.FEDPROX
        self.mu = 0.01  # Proximal term coefficient
    
    def aggregate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate updates using FedProx."""
        self.logger.info("Aggregating updates using FedProx")
        
        if not client_updates:
            return {}
        
        # Use FedAvg as base aggregation
        fedavg_manager = FedAvgManager(self.config)
        fedavg_manager.client_configs = self.client_configs
        aggregated_updates = fedavg_manager.aggregate_updates(client_updates)
        
        # Add proximal term
        if self.global_model is not None:
            for param_name, aggregated_param in aggregated_updates.items():
                if param_name in dict(self.global_model.named_parameters()):
                    global_param = dict(self.global_model.named_parameters())[param_name]
                    # Add proximal term: aggregated_param + mu * (aggregated_param - global_param)
                    proximal_term = self.mu * (aggregated_param - global_param.data)
                    aggregated_updates[param_name] = aggregated_param + proximal_term
        
        return aggregated_updates

class FedNovaManager(BaseFederatedManager):
    """FedNova manager with normalized averaging."""
    
    def __init__(self, config: FederatedConfig):
        super().__init__(config)
        self.strategy = FederatedStrategy.FEDNOVA
        self.client_epoch_counts: Dict[str, int] = {}
    
    def aggregate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate updates using FedNova."""
        self.logger.info("Aggregating updates using FedNova")
        
        if not client_updates:
            return {}
        
        # Normalize updates by local epoch count
        normalized_updates = {}
        
        for client_id, updates in client_updates.items():
            client_config = self.client_configs[client_id]
            local_epochs = client_config.local_epochs
            
            # Normalize by local epochs
            normalized_client_updates = {}
            for param_name, param_update in updates.items():
                normalized_client_updates[param_name] = param_update / local_epochs
            
            normalized_updates[client_id] = normalized_client_updates
        
        # Use FedAvg on normalized updates
        fedavg_manager = FedAvgManager(self.config)
        fedavg_manager.client_configs = self.client_configs
        aggregated_updates = fedavg_manager.aggregate_updates(normalized_updates)
        
        return aggregated_updates

class SecureAggregationManager(BaseFederatedManager):
    """Secure aggregation manager with privacy protection."""
    
    def __init__(self, config: FederatedConfig):
        super().__init__(config)
        self.strategy = FederatedStrategy.SECURE_AGGREGATION
        self.enable_differential_privacy = config.enable_differential_privacy
        self.noise_multiplier = 1.0
    
    def aggregate_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Aggregate updates with secure aggregation."""
        self.logger.info("Aggregating updates using secure aggregation")
        
        if not client_updates:
            return {}
        
        # Use FedAvg as base aggregation
        fedavg_manager = FedAvgManager(self.config)
        fedavg_manager.client_configs = self.client_configs
        aggregated_updates = fedavg_manager.aggregate_updates(client_updates)
        
        # Add differential privacy noise
        if self.enable_differential_privacy:
            for param_name, aggregated_param in aggregated_updates.items():
                # Add Gaussian noise for differential privacy
                noise = torch.randn_like(aggregated_param) * self.noise_multiplier
                aggregated_updates[param_name] = aggregated_param + noise
        
        return aggregated_updates

class TruthGPTFederatedManager:
    """TruthGPT Federated Learning Manager."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.federated_manager = self._create_federated_manager()
        self.training_history: List[Dict[str, Any]] = []
    
    def _create_federated_manager(self) -> BaseFederatedManager:
        """Create federated manager based on strategy."""
        if self.config.strategy == FederatedStrategy.FEDAVG:
            return FedAvgManager(self.config)
        elif self.config.strategy == FederatedStrategy.FEDPROX:
            return FedProxManager(self.config)
        elif self.config.strategy == FederatedStrategy.FEDNOVA:
            return FedNovaManager(self.config)
        elif self.config.strategy == FederatedStrategy.SECURE_AGGREGATION:
            return SecureAggregationManager(self.config)
        else:
            return FedAvgManager(self.config)  # Default
    
    def initialize_federated_learning(
        self,
        global_model: nn.Module,
        client_configs: List[ClientConfig]
    ):
        """Initialize federated learning."""
        self.logger.info("Initializing federated learning")
        
        # Initialize clients
        self.federated_manager.initialize_clients(client_configs)
        
        # Distribute initial model
        self.federated_manager.distribute_model(global_model)
        
        self.logger.info(f"Federated learning initialized with {len(client_configs)} clients")
    
    def run_federated_training(self, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Run federated training."""
        num_rounds = num_rounds or self.config.num_rounds
        
        self.logger.info(f"Starting federated training for {num_rounds} rounds")
        
        start_time = time.time()
        
        for round_num in range(num_rounds):
            # Run federated round
            round_metrics = self.federated_manager.run_federated_round()
            self.training_history.append(round_metrics)
            
            if round_num % 10 == 0:
                self.logger.info(f"Completed round {round_num + 1}/{num_rounds}")
        
        training_time = time.time() - start_time
        
        # Final metrics
        final_metrics = {
            'total_rounds': num_rounds,
            'training_time': training_time,
            'average_round_time': training_time / num_rounds,
            'total_participants': len(self.federated_manager.client_configs),
            'strategy': self.config.strategy.value
        }
        
        self.logger.info(f"Federated training completed in {training_time:.2f}s")
        
        return final_metrics
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get federated training history."""
        return self.training_history.copy()
    
    def get_global_model(self) -> Optional[nn.Module]:
        """Get current global model."""
        return self.federated_manager.global_model
    
    def get_federated_statistics(self) -> Dict[str, Any]:
        """Get federated learning statistics."""
        if not self.training_history:
            return {}
        
        return {
            'total_rounds': len(self.training_history),
            'average_participants': sum(r['num_participants'] for r in self.training_history) / len(self.training_history),
            'strategy': self.config.strategy.value,
            'aggregation_method': self.config.aggregation_method.value,
            'num_clients': len(self.federated_manager.client_configs)
        }

# Factory functions
def create_federated_manager(config: FederatedConfig) -> TruthGPTFederatedManager:
    """Create federated manager."""
    return TruthGPTFederatedManager(config)

def create_fedavg_manager(config: FederatedConfig) -> FedAvgManager:
    """Create FedAvg manager."""
    config.strategy = FederatedStrategy.FEDAVG
    return FedAvgManager(config)

def create_fedprox_manager(config: FederatedConfig) -> FedProxManager:
    """Create FedProx manager."""
    config.strategy = FederatedStrategy.FEDPROX
    return FedProxManager(config)

def create_fednova_manager(config: FederatedConfig) -> FedNovaManager:
    """Create FedNova manager."""
    config.strategy = FederatedStrategy.FEDNOVA
    return FedNovaManager(config)

def create_secure_aggregation(config: FederatedConfig) -> SecureAggregationManager:
    """Create secure aggregation manager."""
    config.strategy = FederatedStrategy.SECURE_AGGREGATION
    return SecureAggregationManager(config)


