"""
Advanced AI-Driven Routing System
=================================

Next-generation routing with reinforcement learning, quantum-inspired algorithms,
and adaptive intelligence.
"""

from .blockchain_router import BlockchainRouter, BlockchainRouterConfig
from .federated_router import FederatedRouter, FederatedRouterConfig
from .quantum_router import QuantumRouter, QuantumRouterConfig
from .reinforcement_router import ReinforcementRouter, ReinforcementRouterConfig

__all__ = [
    # Reinforcement Learning Router
    'ReinforcementRouter',
    'ReinforcementRouterConfig',

    # Quantum-Inspired Router
    'QuantumRouter',
    'QuantumRouterConfig',

    # Federated Learning Router
    'FederatedRouter',
    'FederatedRouterConfig',

    # Blockchain Router
    'BlockchainRouter',
    'BlockchainRouterConfig',
]

