"""
Federated Learning Package
==========================

Advanced federated learning systems with privacy preservation and distributed optimization.
"""
from typing import Any

from .enums import AggregationMethod, ClientSelectionStrategy, PrivacyLevel
from .config import FederatedLearningConfig
from .client import FederatedClient
from .server import FederatedServer, AsyncFederatedServer
from .privacy import PrivacyPreservation
from .system import FederatedLearningSystem

# Compatibility aliases
FederatedConfig = FederatedLearningConfig

# Factory functions
def create_federated_config(**kwargs) -> FederatedLearningConfig:
    return FederatedLearningConfig(**kwargs)

def create_federated_client(client_id: str, model: Any, config: FederatedLearningConfig) -> FederatedClient:
    return FederatedClient(client_id, model, config)

def create_federated_server(global_model: Any, config: FederatedLearningConfig) -> FederatedServer:
    return FederatedServer(global_model, config)

def create_async_federated_server(global_model: Any, config: FederatedLearningConfig) -> AsyncFederatedServer:
    return AsyncFederatedServer(global_model, config)

def create_privacy_preservation(config: FederatedLearningConfig) -> PrivacyPreservation:
    return PrivacyPreservation(config)

def create_federated_learning_system(config: FederatedLearningConfig) -> FederatedLearningSystem:
    return FederatedLearningSystem(config)

__all__ = [
    'AggregationMethod',
    'ClientSelectionStrategy',
    'PrivacyLevel',
    'FederatedLearningConfig',
    'FederatedClient',
    'FederatedServer',
    'AsyncFederatedServer',
    'PrivacyPreservation',
    'FederatedLearningSystem',
    'create_federated_config',
    'create_federated_client',
    'create_federated_server',
    'create_async_federated_server',
    'create_privacy_preservation',
    'create_federated_learning_system'
]

