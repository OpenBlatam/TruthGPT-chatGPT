"""
Neural Architecture Search (NAS) for TruthGPT
Following deep learning best practices for automated architecture optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import random
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class NASConfig:
    """Neural Architecture Search configuration"""
    search_space_size: int = 1000
    population_size: int = 50
    num_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    max_layers: int = 20
    min_layers: int = 4
    hidden_size_range: Tuple[int, int] = (256, 1024)
    num_heads_range: Tuple[int, int] = (4, 16)
    dropout_range: Tuple[float, float] = (0.1, 0.5)


class ArchitectureGene:
    """Single architecture gene representing a layer configuration"""
    
    def __init__(self, layer_type: str, hidden_size: int, num_heads: int, 
                 dropout: float, activation: str):
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
    
    def mutate(self, config: NASConfig) -> 'ArchitectureGene':
        """Mutate this gene"""
        new_gene = ArchitectureGene(
            layer_type=self.layer_type,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation
        )
        
        # Random mutations
        if random.random() < config.mutation_rate:
            new_gene.layer_type = random.choice(['attention', 'feedforward', 'normalization'])
        
        if random.random() < config.mutation_rate:
            new_gene.hidden_size = random.randint(*config.hidden_size_range)
        
        if random.random() < config.mutation_rate:
            new_gene.num_heads = random.randint(*config.num_heads_range)
        
        if random.random() < config.mutation_rate:
            new_gene.dropout = random.uniform(*config.dropout_range)
        
        if random.random() < config.mutation_rate:
            new_gene.activation = random.choice(['gelu', 'relu', 'swish', 'mish'])
        
        return new_gene
    
    def crossover(self, other: 'ArchitectureGene') -> Tuple['ArchitectureGene', 'ArchitectureGene']:
        """Crossover with another gene"""
        child1 = ArchitectureGene(
            layer_type=self.layer_type if random.random() < 0.5 else other.layer_type,
            hidden_size=self.hidden_size if random.random() < 0.5 else other.hidden_size,
            num_heads=self.num_heads if random.random() < 0.5 else other.num_heads,
            dropout=self.dropout if random.random() < 0.5 else other.dropout,
            activation=self.activation if random.random() < 0.5 else other.activation
        )
        
        child2 = ArchitectureGene(
            layer_type=other.layer_type if random.random() < 0.5 else self.layer_type,
            hidden_size=other.hidden_size if random.random() < 0.5 else self.hidden_size,
            num_heads=other.num_heads if random.random() < 0.5 else self.num_heads,
            dropout=other.dropout if random.random() < 0.5 else self.dropout,
            activation=other.activation if random.random() < 0.5 else self.activation
        )
        
        return child1, child2


class ArchitectureChromosome:
    """Complete architecture represented as a chromosome"""
    
    def __init__(self, genes: List[ArchitectureGene]):
        self.genes = genes
        self.fitness = 0.0
        self.performance_metrics = {}
    
    def mutate(self, config: NASConfig) -> 'ArchitectureChromosome':
        """Mutate this chromosome"""
        new_genes = []
        
        for gene in self.genes:
            if random.random() < config.mutation_rate:
                new_genes.append(gene.mutate(config))
            else:
                new_genes.append(gene)
        
        # Add or remove layers
        if random.random() < config.mutation_rate and len(new_genes) < config.max_layers:
            # Add new layer
            new_gene = ArchitectureGene(
                layer_type=random.choice(['attention', 'feedforward', 'normalization']),
                hidden_size=random.randint(*config.hidden_size_range),
                num_heads=random.randint(*config.num_heads_range),
                dropout=random.uniform(*config.dropout_range),
                activation=random.choice(['gelu', 'relu', 'swish', 'mish'])
            )
            new_genes.append(new_gene)
        
        if random.random() < config.mutation_rate and len(new_genes) > config.min_layers:
            # Remove random layer
            new_genes.pop(random.randint(0, len(new_genes) - 1))
        
        return ArchitectureChromosome(new_genes)
    
    def crossover(self, other: 'ArchitectureChromosome') -> Tuple['ArchitectureChromosome', 'ArchitectureChromosome']:
        """Crossover with another chromosome"""
        min_len = min(len(self.genes), len(other.genes))
        max_len = max(len(self.genes), len(other.genes))
        
        # Random crossover point
        crossover_point = random.randint(1, min_len - 1) if min_len > 1 else 1
        
        # Create children
        child1_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child2_genes = other.genes[:crossover_point] + self.genes[crossover_point:]
        
        # Handle length differences
        if len(self.genes) > len(other.genes):
            child1_genes.extend(self.genes[crossover_point:])
        elif len(other.genes) > len(self.genes):
            child2_genes.extend(other.genes[crossover_point:])
        
        return ArchitectureChromosome(child1_genes), ArchitectureChromosome(child2_genes)
    
    def to_model(self, vocab_size: int, max_length: int) -> nn.Module:
        """Convert chromosome to PyTorch model"""
        return ArchitectureToModel(self, vocab_size, max_length)


class ArchitectureToModel(nn.Module):
    """Convert architecture chromosome to PyTorch model"""
    
    def __init__(self, chromosome: ArchitectureChromosome, vocab_size: int, max_length: int):
        super().__init__()
        self.chromosome = chromosome
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Build model from genes
        self.layers = nn.ModuleList()
        self._build_model()
    
    def _build_model(self):
        """Build model from chromosome genes"""
        # Input embedding
        self.embedding = nn.Embedding(self.vocab_size, self.chromosome.genes[0].hidden_size)
        self.pos_embedding = nn.Embedding(self.max_length, self.chromosome.genes[0].hidden_size)
        
        # Build layers from genes
        for gene in self.chromosome.genes:
            if gene.layer_type == 'attention':
                layer = MultiHeadAttention(
                    hidden_size=gene.hidden_size,
                    num_heads=gene.num_heads,
                    dropout=gene.dropout
                )
            elif gene.layer_type == 'feedforward':
                layer = FeedForward(
                    hidden_size=gene.hidden_size,
                    dropout=gene.dropout,
                    activation=gene.activation
                )
            elif gene.layer_type == 'normalization':
                layer = nn.LayerNorm(gene.hidden_size)
            else:
                continue
            
            self.layers.append(layer)
        
        # Output layer
        self.output_proj = nn.Linear(self.chromosome.genes[-1].hidden_size, self.vocab_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeddings = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(pos_ids)
        x = token_embeddings + pos_embeddings
        
        # Apply layers
        for layer in self.layers:
            if isinstance(layer, MultiHeadAttention):
                x = layer(x, attention_mask)
            else:
                x = layer(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        return {'logits': logits}


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-forward layer"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        
        self.linear1 = nn.Linear(hidden_size, self.intermediate_size)
        self.linear2 = nn.Linear(self.intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        else:
            self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor):
        """Forward pass"""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class NASOptimizer:
    """Neural Architecture Search optimizer"""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_architecture = None
        self.fitness_history = []
    
    def initialize_population(self):
        """Initialize random population"""
        self.population = []
        
        for _ in range(self.config.population_size):
            # Random number of layers
            num_layers = random.randint(self.config.min_layers, self.config.max_layers)
            
            # Create random genes
            genes = []
            for _ in range(num_layers):
                gene = ArchitectureGene(
                    layer_type=random.choice(['attention', 'feedforward', 'normalization']),
                    hidden_size=random.randint(*self.config.hidden_size_range),
                    num_heads=random.randint(*self.config.num_heads_range),
                    dropout=random.uniform(*self.config.dropout_range),
                    activation=random.choice(['gelu', 'relu', 'swish', 'mish'])
                )
                genes.append(gene)
            
            chromosome = ArchitectureChromosome(genes)
            self.population.append(chromosome)
    
    def evaluate_population(self, eval_fn: callable):
        """Evaluate population fitness"""
        for chromosome in self.population:
            if chromosome.fitness == 0.0:  # Not evaluated yet
                try:
                    # Convert to model and evaluate
                    model = chromosome.to_model(vocab_size=50257, max_length=512)
                    fitness = eval_fn(model)
                    chromosome.fitness = fitness
                except Exception as e:
                    # Invalid architecture
                    chromosome.fitness = -float('inf')
    
    def selection(self) -> List[ArchitectureChromosome]:
        """Select parents for next generation"""
        # Sort by fitness
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # Elite selection
        elite_size = int(self.config.population_size * self.config.elite_ratio)
        elite = sorted_population[:elite_size]
        
        # Tournament selection for rest
        parents = elite.copy()
        
        while len(parents) < self.config.population_size:
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def crossover_and_mutation(self, parents: List[ArchitectureChromosome]) -> List[ArchitectureChromosome]:
        """Create new generation through crossover and mutation"""
        new_population = []
        
        # Keep elite
        elite_size = int(self.config.population_size * self.config.elite_ratio)
        new_population.extend(parents[:elite_size])
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            if random.random() < self.config.crossover_rate:
                # Crossover
                child1, child2 = parent1.crossover(parent2)
            else:
                # No crossover
                child1, child2 = parent1, parent2
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = child1.mutate(self.config)
            if random.random() < self.config.mutation_rate:
                child2 = child2.mutate(self.config)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:self.config.population_size]
    
    def evolve(self, eval_fn: callable, num_generations: Optional[int] = None):
        """Evolve architecture population"""
        if num_generations is None:
            num_generations = self.config.num_generations
        
        # Initialize population
        if not self.population:
            self.initialize_population()
        
        for generation in range(num_generations):
            self.generation = generation
            
            # Evaluate population
            self.evaluate_population(eval_fn)
            
            # Track best
            best_chromosome = max(self.population, key=lambda x: x.fitness)
            if self.best_architecture is None or best_chromosome.fitness > self.best_architecture.fitness:
                self.best_architecture = best_chromosome
            
            # Record fitness
            avg_fitness = np.mean([c.fitness for c in self.population])
            self.fitness_history.append(avg_fitness)
            
            # Selection
            parents = self.selection()
            
            # Crossover and mutation
            self.population = self.crossover_and_mutation(parents)
            
            print(f"Generation {generation + 1}: Best fitness = {best_chromosome.fitness:.4f}, "
                  f"Avg fitness = {avg_fitness:.4f}")
    
    def get_best_architecture(self) -> ArchitectureChromosome:
        """Get best architecture found"""
        return self.best_architecture
    
    def get_search_results(self) -> Dict[str, Any]:
        """Get search results and statistics"""
        return {
            'best_fitness': self.best_architecture.fitness if self.best_architecture else 0.0,
            'fitness_history': self.fitness_history,
            'generation': self.generation,
            'population_size': len(self.population),
            'best_architecture': self.best_architecture
        }



