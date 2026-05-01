"""
Neural Architecture Search Package
==================================

Automated neural architecture discovery and optimization.
"""
from .enums import SearchStrategy
from .config import NASConfig
from .gene import ArchitectureGene
from .architecture import NeuralArchitecture
from .evolutionary import EvolutionaryNAS
from .differentiable import DifferentiableNAS

# Compatibility aliases
NeuralArchitectureSearch = EvolutionaryNAS

# Factory functions
def create_nas_config(**kwargs) -> NASConfig:
    return NASConfig(**kwargs)

def create_evolutionary_nas(config: NASConfig) -> EvolutionaryNAS:
    return EvolutionaryNAS(config)

def create_differentiable_nas(config: NASConfig) -> DifferentiableNAS:
    return DifferentiableNAS(config)

__all__ = [
    'SearchStrategy',
    'NASConfig',
    'ArchitectureGene',
    'NeuralArchitecture',
    'EvolutionaryNAS',
    'DifferentiableNAS',
    'create_nas_config',
    'create_evolutionary_nas',
    'create_differentiable_nas'
]

