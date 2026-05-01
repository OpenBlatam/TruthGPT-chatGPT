"""
Architecture Gene
=================

Gene representation for a neural network layer.
"""
import torch.nn as nn
import random
from typing import Dict, Any, Tuple
from .config import NASConfig

class ArchitectureGene:
    """Gene representing a layer in neural architecture"""
    
    def __init__(self, layer_type: str, params: Dict[str, Any]):
        self.layer_type = layer_type
        self.params = params
        self.fitness = 0.0
    
    def mutate(self, config: NASConfig):
        """Mutate this gene"""
        # Mutate layer type
        if random.random() < config.mutation_rate:
            self.layer_type = random.choice(config.layer_types)
        
        # Mutate parameters
        for param_name, param_value in self.params.items():
            if random.random() < config.mutation_rate:
                if isinstance(param_value, int):
                    # Mutate integer parameters
                    if param_name in ['out_features', 'hidden_size', 'embed_dim']:
                        self.params[param_name] = random.randint(
                            config.min_width, config.max_width
                        )
                    elif param_name == 'num_heads':
                        self.params[param_name] = random.choice([4, 8, 16, 32])
                elif isinstance(param_value, float):
                    # Mutate float parameters
                    self.params[param_name] = max(0.0, param_value + random.gauss(0, 0.1))
    
    def crossover(self, other: 'ArchitectureGene') -> Tuple['ArchitectureGene', 'ArchitectureGene']:
        """Crossover with another gene"""
        # Create offspring
        child1 = ArchitectureGene(self.layer_type, self.params.copy())
        child2 = ArchitectureGene(other.layer_type, other.params.copy())
        
        # Crossover parameters
        for param_name in self.params:
            if random.random() < 0.5:
                child1.params[param_name] = other.params[param_name]
                child2.params[param_name] = self.params[param_name]
        
        return child1, child2
    
    def to_layer(self) -> nn.Module:
        """Convert gene to PyTorch layer"""
        if self.layer_type == 'Linear':
            return nn.Linear(
                self.params.get('in_features', 128),
                self.params.get('out_features', 64)
            )
        elif self.layer_type == 'Conv2d':
            return nn.Conv2d(
                self.params.get('in_channels', 3),
                self.params.get('out_channels', 32),
                kernel_size=self.params.get('kernel_size', 3),
                stride=self.params.get('stride', 1),
                padding=self.params.get('padding', 1)
            )
        elif self.layer_type == 'LSTM':
            return nn.LSTM(
                input_size=self.params.get('input_size', 128),
                hidden_size=self.params.get('hidden_size', 64),
                num_layers=self.params.get('num_layers', 1),
                batch_first=True
            )
        elif self.layer_type == 'MultiheadAttention':
            return nn.MultiheadAttention(
                embed_dim=self.params.get('embed_dim', 128),
                num_heads=self.params.get('num_heads', 8),
                dropout=self.params.get('dropout', 0.1)
            )
        else:
            # Default to Linear
            return nn.Linear(128, 64)

