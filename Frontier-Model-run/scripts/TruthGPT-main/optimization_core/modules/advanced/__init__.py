"""
Ultra-Advanced TruthGPT Optimization Modules
Following deep learning best practices for maximum performance
"""

from .quantum_optimization import QuantumOptimizer, QuantumAttention, QuantumLayerNorm
from .neural_architecture_search import NASOptimizer

__all__ = [
    # Quantum optimization
    'QuantumOptimizer', 'QuantumAttention', 'QuantumLayerNorm',
    
    # Neural Architecture Search
    'NASOptimizer',
    
    # Distributed training
    'DistributedTrainer', 'HorovodTrainer', 'RayTrainer',
    
    # Model compression
    'ModelCompressor', 'PruningOptimizer', 'KnowledgeDistillation',
]



