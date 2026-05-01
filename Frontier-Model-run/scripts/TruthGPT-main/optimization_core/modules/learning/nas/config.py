"""
NAS Configuration
=================

Configuration for neural architecture search systems.
"""
from dataclasses import dataclass, field
from typing import List
from .enums import SearchStrategy

@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search"""
    # Search parameters
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Architecture constraints
    max_layers: int = 20
    min_layers: int = 3
    max_width: int = 1024
    min_width: int = 32
    
    # Search space
    layer_types: List[str] = field(default_factory=lambda: [
        'Linear', 'Conv2d', 'LSTM', 'GRU', 'MultiheadAttention', 'Transformer'
    ])
    activation_types: List[str] = field(default_factory=lambda: [
        'ReLU', 'GELU', 'Swish', 'Mish', 'LeakyReLU'
    ])
    
    # Optimization
    search_budget: int = 1000  # Total evaluations
    early_stopping_patience: int = 20
    performance_threshold: float = 0.95
    
    # Advanced features
    enable_multi_objective: bool = True
    enable_transfer_learning: bool = True
    enable_architecture_pruning: bool = True
    
    def __post_init__(self):
        """Validate NAS configuration"""
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if self.mutation_rate < 0.0 or self.mutation_rate > 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")

