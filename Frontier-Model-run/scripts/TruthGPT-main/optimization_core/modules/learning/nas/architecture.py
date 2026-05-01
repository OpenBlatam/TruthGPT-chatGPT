"""
Neural Architecture
===================

Neural architecture representation as a sequence of genes.
"""
import torch.nn as nn
import random
from typing import List, Tuple, Dict, Any
from .config import NASConfig
from .gene import ArchitectureGene

class NeuralArchitecture:
    """Neural architecture represented as a sequence of genes"""
    
    def __init__(self, genes: List[ArchitectureGene]):
        self.genes = genes
        self.fitness = 0.0
        self.complexity = self._calculate_complexity()
        self.performance_metrics = {}
    
    def _calculate_complexity(self) -> float:
        """Calculate architecture complexity"""
        total_params = 0
        for gene in self.genes:
            if gene.layer_type == 'Linear':
                in_features = gene.params.get('in_features', 128)
                out_features = gene.params.get('out_features', 64)
                total_params += in_features * out_features
            elif gene.layer_type == 'Conv2d':
                in_channels = gene.params.get('in_channels', 3)
                out_channels = gene.params.get('out_channels', 32)
                kernel_size = gene.params.get('kernel_size', 3)
                total_params += in_channels * out_channels * kernel_size * kernel_size
        
        return total_params / 1e6  # Convert to millions
    
    def mutate(self, config: NASConfig):
        """Mutate architecture"""
        # Mutate individual genes
        for gene in self.genes:
            gene.mutate(config)
        
        # Add/remove layers
        if random.random() < config.mutation_rate:
            if len(self.genes) < config.max_layers and random.random() < 0.5:
                # Add new layer
                new_gene = ArchitectureGene(
                    random.choice(config.layer_types),
                    self._generate_random_params(random.choice(config.layer_types))
                )
                insert_pos = random.randint(0, len(self.genes))
                self.genes.insert(insert_pos, new_gene)
            elif len(self.genes) > config.min_layers:
                # Remove layer
                remove_pos = random.randint(0, len(self.genes) - 1)
                del self.genes[remove_pos]
        
        # Update complexity
        self.complexity = self._calculate_complexity()
    
    def crossover(self, other: 'NeuralArchitecture') -> Tuple['NeuralArchitecture', 'NeuralArchitecture']:
        """Crossover with another architecture"""
        # Single-point crossover
        crossover_point = random.randint(1, min(len(self.genes), len(other.genes)) - 1)
        
        child1_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child2_genes = other.genes[:crossover_point] + self.genes[crossover_point:]
        
        return NeuralArchitecture(child1_genes), NeuralArchitecture(child2_genes)
    
    def to_model(self, input_size: int = 128, output_size: int = 10) -> nn.Module:
        """Convert architecture to PyTorch model"""
        layers = []
        
        # Input layer
        current_size = input_size
        
        for i, gene in enumerate(self.genes):
            layer = gene.to_layer()
            
            # Adjust input size for Linear layers
            if isinstance(layer, nn.Linear):
                layer.in_features = current_size
                current_size = layer.out_features
            
            layers.append(layer)
            
            # Add activation
            activation = random.choice(['ReLU', 'GELU', 'Swish'])
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'GELU':
                layers.append(nn.GELU())
            elif activation == 'Swish':
                layers.append(nn.SiLU())  # Swish is SiLU in PyTorch
            
            # Add dropout
            if random.random() < 0.3:
                layers.append(nn.Dropout(0.1))
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        return nn.Sequential(*layers)
    
    def _generate_random_params(self, layer_type: str) -> Dict[str, Any]:
        """Generate random parameters for layer type"""
        if layer_type == 'Linear':
            return {
                'in_features': random.randint(32, 512),
                'out_features': random.randint(32, 512)
            }
        elif layer_type == 'Conv2d':
            return {
                'in_channels': random.choice([1, 3, 16, 32]),
                'out_channels': random.choice([16, 32, 64, 128]),
                'kernel_size': random.choice([3, 5, 7]),
                'stride': random.choice([1, 2]),
                'padding': random.choice([0, 1, 2])
            }
        elif layer_type == 'LSTM':
            return {
                'input_size': random.randint(32, 256),
                'hidden_size': random.randint(32, 256),
                'num_layers': random.randint(1, 3)
            }
        elif layer_type == 'MultiheadAttention':
            return {
                'embed_dim': random.choice([64, 128, 256, 512]),
                'num_heads': random.choice([4, 8, 16]),
                'dropout': random.uniform(0.0, 0.3)
            }
        else:
            return {}

