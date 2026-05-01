"""
Advanced PiMoE Routing Algorithms
=================================

Enhanced routing mechanisms with attention-based routing, dynamic expert scaling,
and cross-expert communication for improved performance and flexibility.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimization_core.modules.feed_forward.routing.pimoe_router import (
    ExpertType,
    PiMoEExpert,
    RoutingDecision,
)


class RoutingStrategy(Enum):
    """Advanced routing strategies."""
    ATTENTION_BASED = "attention_based"
    HIERARCHICAL = "hierarchical"
    DYNAMIC_SCALING = "dynamic_scaling"
    CROSS_EXPERT = "cross_expert"
    ADAPTIVE_LEARNING = "adaptive_learning"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"


@dataclass
class AdvancedRoutingConfig:
    """
    Configuration for advanced routing algorithms.

    Attributes:
        strategy: Routing strategy to use.
        attention_heads: Number of attention heads.
        hierarchical_levels: Number of levels in hierarchical routing.
        dynamic_scaling_threshold: Threshold for dynamic scaling.
        cross_expert_communication: Whether to enable cross-expert communication.
        adaptive_learning_rate: Learning rate for adaptive mechanisms.
        nas_search_space: Size of NAS search space.
        temperature_schedule: Schedule for temperature (cosine, linear, etc.).
        load_balance_alpha: Alpha parameter for load balancing.
        expert_capacity_factor: Capacity factor for experts.
        routing_entropy_weight: Weight for routing entropy regularization.
    """
    strategy: RoutingStrategy = RoutingStrategy.ATTENTION_BASED
    attention_heads: int = 8
    hierarchical_levels: int = 3
    dynamic_scaling_threshold: float = 0.8
    cross_expert_communication: bool = True
    adaptive_learning_rate: float = 0.01
    nas_search_space: int = 100
    temperature_schedule: str = "cosine"  # cosine, linear, exponential
    load_balance_alpha: float = 0.1
    expert_capacity_factor: float = 1.5
    routing_entropy_weight: float = 0.05


class AttentionBasedRouter(nn.Module):
    """
    Attention-based router for more sophisticated token-level routing decisions.
    Uses multi-head attention to determine expert assignments.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_types: List[ExpertType],
        attention_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0
    ) -> None:
        """
        Initialize AttentionBasedRouter.

        Args:
            hidden_size: Hidden dimension size.
            num_experts: Number of experts.
            expert_types: List of expert types.
            attention_heads: Number of attention heads.
            dropout: Dropout probability.
            temperature: Temperature for routing softmax.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_types = expert_types
        self.attention_heads = attention_heads
        self.head_dim = hidden_size // attention_heads
        self.dropout = dropout
        self.temperature = temperature

        # Multi-head attention for routing
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)

        # Expert-specific attention
        self.expert_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.head_dim,
                num_heads=1,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_experts)
        ])

        # Expert type embeddings
        self.expert_type_embeddings = nn.Embedding(len(expert_types), hidden_size)

        # Routing decision network
        self.routing_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_experts)
        )

        # Load balancing
        self.register_buffer('expert_loads', torch.zeros(num_experts))
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with attention-based routing.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Attention mask.
            return_attention_weights: Whether to return internal attention weights.

        Returns:
            Routed hidden states or tuple with info.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Use routing network to determine expert assignments
        # [batch, seq, hidden] -> [batch, seq, num_experts]
        combined_attention = self.routing_network(hidden_states)

        # Apply temperature scaling
        routing_scores = combined_attention / self.temperature

        # Get routing decisions
        expert_probs = F.softmax(routing_scores, dim=-1)
        top_expert_scores, top_expert_indices = torch.topk(expert_probs, k=1, dim=-1)

        # Create routing decisions
        routing_decisions: List[RoutingDecision] = []
        # Note: Optimization needed for large batch/seq_len
        if return_attention_weights:
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    expert_id = top_expert_indices[batch_idx, seq_idx].item()
                    expert_type = self.expert_types[expert_id % len(self.expert_types)]

                    decision = RoutingDecision(
                        token_id=batch_idx * seq_len + seq_idx,
                        expert_id=expert_id,
                        expert_type=expert_type,
                        confidence=top_expert_scores[batch_idx, seq_idx].item(),
                        routing_score=routing_scores[batch_idx, seq_idx, expert_id].item(),
                        load_balance_weight=self._calculate_load_balance_weight(expert_id)
                    )
                    routing_decisions.append(decision)

        # Update expert usage
        self._update_expert_usage(top_expert_indices)

        # Apply expert routing
        routed_hidden = self._apply_expert_routing(
            hidden_states, top_expert_indices, top_expert_scores
        )

        if return_attention_weights:
            attention_info = {
                'routing_decisions': routing_decisions,
                'expert_probs': expert_probs,
                'attention_weights': combined_attention,
                'load_balance_loss': self._calculate_load_balance_loss(expert_probs)
            }
            return routed_hidden, attention_info

        return routed_hidden

    def _calculate_load_balance_weight(self, expert_id: int) -> float:
        """Calculate load balancing weight for an expert."""
        count = self.expert_usage_count[expert_id].item()
        if count == 0:
            return 1.0
        return 1.0 / (1.0 + count)

    def _update_expert_usage(self, expert_indices: torch.Tensor) -> None:
        """Update expert usage statistics."""
        for expert_id in expert_indices.flatten():
            self.expert_usage_count[expert_id] += 1
            self.expert_loads[expert_id] += 1

    def _apply_expert_routing(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_scores: torch.Tensor
    ) -> torch.Tensor:
        """Apply expert routing with attention-based processing."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Create expert-specific transformations
        expert_weights = F.one_hot(expert_indices.squeeze(-1), num_classes=self.num_experts).float()

        # Expert-specific attention processing
        routed_hidden = torch.zeros_like(hidden_states)

        for expert_id in range(self.num_experts):
            expert_mask = expert_weights[:, :, expert_id:expert_id+1]
            expert_input = hidden_states * expert_mask

            # Apply expert-specific processing
            expert_output = self._process_expert_input(expert_input, expert_id)
            routed_hidden += expert_output * expert_mask

        return routed_hidden

    def _process_expert_input(self, expert_input: torch.Tensor, expert_id: int) -> torch.Tensor:
        """Process input for a specific expert."""
        # Expert-specific transformation (Simulation)
        expert_transform = torch.randn(self.hidden_size, self.hidden_size, device=expert_input.device)
        expert_bias = torch.randn(self.hidden_size, device=expert_input.device)
        
        # In standardized implementation, use matmul
        return torch.matmul(expert_input, expert_transform) + expert_bias

    def _calculate_load_balance_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """Calculate load balancing auxiliary loss."""
        # Average over batch and sequence
        expert_usage = torch.mean(expert_probs, dim=[0, 1])
        uniform_usage = torch.ones_like(expert_usage) / self.num_experts

        load_balance_loss = F.mse_loss(expert_usage, uniform_usage)
        return load_balance_loss

    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """Get expert usage statistics."""
        return {
            'expert_usage_counts': self.expert_usage_count.cpu().numpy().tolist(),
            'expert_loads': self.expert_loads.cpu().numpy().tolist(),
        }


class HierarchicalRouter(nn.Module):
    """
    Hierarchical router with multi-level routing decisions.
    Routes tokens through multiple levels of expert selection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_types: List[ExpertType],
        hierarchical_levels: int = 3,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize HierarchicalRouter.

        Args:
            hidden_size: Hidden dimension.
            num_experts: Number of experts.
            expert_types: List of expert types.
            hierarchical_levels: Number of routing levels.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_types = expert_types
        self.hierarchical_levels = hierarchical_levels
        self.dropout = dropout

        # Hierarchical routing networks
        self.level_routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_experts)
            )
            for _ in range(hierarchical_levels)
        ])

        # Level-specific expert assignments
        self.level_expert_assignments = nn.Parameter(
            torch.randn(hierarchical_levels, num_experts)
        )

        # Hierarchical combination weights
        self.hierarchical_weights = nn.Parameter(
            torch.ones(hierarchical_levels)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hierarchical_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with hierarchical routing.

        Args:
            hidden_states: Input tensor.
            attention_mask: Attention mask.
            return_hierarchical_info: Whether to return hierarchical details.

        Returns:
            Output tensor or tuple.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Process through each hierarchical level
        level_outputs = []
        level_decisions = []

        current_hidden = hidden_states

        for level in range(self.hierarchical_levels):
            # Get routing scores for this level
            level_scores = self.level_routers[level](current_hidden)
            level_probs = F.softmax(level_scores, dim=-1)

            # Get top expert for this level
            top_scores, top_indices = torch.topk(level_probs, k=1, dim=-1)

            # Apply expert processing for this level
            level_output = self._apply_level_expert_processing(
                current_hidden, top_indices, top_scores, level
            )

            level_outputs.append(level_output)
            level_decisions.append({
                'level': level,
                'expert_indices': top_indices,
                'expert_scores': top_scores,
                'expert_probs': level_probs
            })

            # Update hidden states for next level
            current_hidden = level_output

        # Combine hierarchical outputs
        hierarchical_weights = F.softmax(self.hierarchical_weights, dim=0)
        final_output = torch.zeros_like(hidden_states)

        for level, output in enumerate(level_outputs):
            final_output += hierarchical_weights[level] * output

        if return_hierarchical_info:
            hierarchical_info = {
                'level_outputs': level_outputs,
                'level_decisions': level_decisions,
                'hierarchical_weights': hierarchical_weights,
                'final_output': final_output
            }
            return final_output, hierarchical_info

        return final_output

    def _apply_level_expert_processing(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_scores: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """Apply expert processing for a specific hierarchical level."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get level-specific expert assignments
        level_assignments = self.level_expert_assignments[level]

        # Create expert masks
        expert_masks = F.one_hot(expert_indices.squeeze(-1), num_classes=self.num_experts).float()

        # Apply level-specific processing
        level_output = torch.zeros_like(hidden_states)

        for expert_id in range(self.num_experts):
            expert_mask = expert_masks[:, :, expert_id:expert_id+1]
            expert_input = hidden_states * expert_mask

            # Level-specific expert transformation
            expert_transform = level_assignments[expert_id] * torch.randn(
                hidden_size, hidden_size, device=hidden_states.device
            )
            expert_bias = level_assignments[expert_id] * torch.randn(
                hidden_size, device=hidden_states.device
            )

            expert_output = torch.matmul(expert_input, expert_transform) + expert_bias
            level_output += expert_output * expert_mask

        return level_output


class DynamicExpertScaler(nn.Module):
    """
    Dynamic expert scaling based on load and performance.
    Automatically adjusts expert capacity based on demand.
    """

    def __init__(
        self,
        base_num_experts: int,
        max_num_experts: int = 16,
        scaling_threshold: float = 0.8,
        scaling_factor: float = 1.5
    ) -> None:
        """
        Initialize DynamicExpertScaler.

        Args:
            base_num_experts: Base number of experts.
            max_num_experts: Maximum number of experts.
            scaling_threshold: Threshold for scaling trigger.
            scaling_factor: Factor to scale by.
        """
        super().__init__()

        self.base_num_experts = base_num_experts
        self.max_num_experts = max_num_experts
        self.scaling_threshold = scaling_threshold
        self.scaling_factor = scaling_factor

        # Expert scaling state
        self.register_buffer('current_num_experts', torch.tensor(base_num_experts))
        self.register_buffer('expert_loads', torch.zeros(max_num_experts))
        self.register_buffer('expert_performance', torch.ones(max_num_experts))

        # Scaling decision network
        self.scaling_network = nn.Sequential(
            nn.Linear(max_num_experts * 2, 64),  # loads + performance
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, expert_loads: torch.Tensor, expert_performance: torch.Tensor) -> Dict[str, Any]:
        """
        Determine if expert scaling is needed.

        Args:
            expert_loads: Information about expert load.
            expert_performance: Information about expert performance.

        Returns:
            Decision dictionary.
        """
        # Prepare input for scaling decision
        scaling_input = torch.cat([expert_loads, expert_performance], dim=-1)
        scaling_decision = self.scaling_network(scaling_input)

        # Determine scaling action
        if scaling_decision > self.scaling_threshold:
            action = "scale_up"
            new_num_experts = min(
                int(self.current_num_experts * self.scaling_factor),
                self.max_num_experts
            )
        elif scaling_decision < (1 - self.scaling_threshold):
            action = "scale_down"
            new_num_experts = max(
                int(self.current_num_experts / self.scaling_factor),
                self.base_num_experts
            )
        else:
            action = "maintain"
            new_num_experts = int(self.current_num_experts.item())

        return {
            'scaling_decision': scaling_decision.item(),
            'action': action,
            'current_experts': self.current_num_experts.item(),
            'new_experts': new_num_experts,
            'scaling_factor': self.scaling_factor
        }

    def update_expert_metrics(self, expert_loads: torch.Tensor, expert_performance: torch.Tensor) -> None:
        """Update expert performance metrics."""
        self.expert_loads = expert_loads
        self.expert_performance = expert_performance


class CrossExpertCommunicator(nn.Module):
    """
    Cross-expert communication mechanism for information sharing.
    Enables experts to share information and collaborate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        communication_channels: int = 4,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize CrossExpertCommunicator.

        Args:
            hidden_size: Hidden dimension.
            num_experts: Number of experts.
            communication_channels: Number of comm channels.
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.communication_channels = communication_channels
        self.dropout = dropout

        # Communication networks
        self.communication_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size)
            )
            for _ in range(communication_channels)
        ])

        # Expert-to-expert attention
        self.expert_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Communication gates
        self.communication_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        expert_ids: List[int],
        return_communication_info: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, Any]]]:
        """
        Enable cross-expert communication.

        Args:
            expert_outputs: List of output tensors from experts.
            expert_ids: List of expert IDs.
            return_communication_info: Whether to return debug info.

        Returns:
            List of tensors or tuple.
        """
        if len(expert_outputs) == 0:
            return expert_outputs

        # Stack expert outputs
        stacked_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden_size]

        # Cross-expert attention
        attended_outputs, attention_weights = self.expert_attention(
            stacked_outputs, stacked_outputs, stacked_outputs
        )

        # Apply communication channels
        communicated_outputs = []
        communication_info: Dict[str, Any] = {
            'attention_weights': attention_weights,
            'communication_channels': []
        }

        for expert_idx, expert_output in enumerate(expert_outputs):
            # Get communication gate
            gate = self.communication_gates[expert_idx](expert_output)

            # Apply communication channels
            channel_outputs = []
            for channel_idx in range(self.communication_channels):
                channel_output = self.communication_networks[channel_idx](expert_output)
                channel_outputs.append(channel_output)

            # Combine channel outputs
            combined_channels = torch.stack(channel_outputs, dim=-1).mean(dim=-1)

            # Apply communication gate
            communicated_output = gate * combined_channels + (1 - gate) * expert_output
            communicated_outputs.append(communicated_output)

            communication_info['communication_channels'].append({
                'expert_id': expert_idx,
                'gate_value': gate.item(),
                'channel_outputs': len(channel_outputs)
            })

        if return_communication_info:
            return communicated_outputs, communication_info

        return communicated_outputs


class NeuralArchitectureSearchRouter(nn.Module):
    """
    Neural Architecture Search for optimal expert configurations.
    Automatically discovers the best expert architectures.
    """

    def __init__(
        self,
        hidden_size: int,
        search_space_size: int = 100,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ) -> None:
        """Initialize NAS Router."""
        super().__init__()

        self.hidden_size = hidden_size
        self.search_space_size = search_space_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Architecture search space
        self.architecture_space = self._initialize_search_space()

        # Population of architectures
        self.population = self._initialize_population()

        # Performance tracking
        self.performance_history: Dict[int, List[float]] = defaultdict(list)

    def _initialize_search_space(self) -> Dict[str, List[Any]]:
        """Initialize the search space for expert architectures."""
        return {
            'num_layers': [1, 2, 3, 4, 5],
            'hidden_sizes': [self.hidden_size // 4, self.hidden_size // 2, self.hidden_size, self.hidden_size * 2],
            'activations': ['relu', 'gelu', 'swish', 'tanh'],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3],
            'normalization': ['layer_norm', 'batch_norm', 'group_norm', 'none']
        }

    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize population of random architectures."""
        population = []

        for _ in range(self.population_size):
            architecture = {
                'num_layers': int(np.random.choice(self.architecture_space['num_layers'])),
                'hidden_sizes': int(np.random.choice(self.architecture_space['hidden_sizes'])),
                'activations': str(np.random.choice(self.architecture_space['activations'])),
                'dropout_rates': float(np.random.choice(self.architecture_space['dropout_rates'])),
                'normalization': str(np.random.choice(self.architecture_space['normalization']))
            }
            population.append(architecture)

        return population

    def evaluate_architecture(self, architecture: Dict[str, Any], performance_metrics: Dict[str, float]) -> float:
        """Evaluate an architecture based on performance metrics."""
        # Multi-objective evaluation
        latency_score = 1.0 / (1.0 + performance_metrics.get('latency_ms', 0))
        throughput_score = performance_metrics.get('throughput_tokens_per_sec', 0) / 1000
        memory_score = 1.0 / (1.0 + performance_metrics.get('memory_usage_mb', 0) / 100)

        # Architecture complexity penalty
        complexity_penalty = (
            architecture['num_layers'] * 0.1 +
            architecture['hidden_sizes'] / self.hidden_size * 0.1
        )

        # Combined fitness score
        fitness = (latency_score * 0.4 + throughput_score * 0.4 + memory_score * 0.2) - complexity_penalty

        return float(fitness)

    def evolve_population(self, performance_data: Dict[Any, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Evolve the population using genetic algorithms."""
        # Evaluate current population
        fitness_scores = []
        for i, architecture in enumerate(self.population):
            if i in performance_data:
                fitness = self.evaluate_architecture(architecture, performance_data[i])
                fitness_scores.append(fitness)
            else:
                fitness_scores.append(0.0)

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]

        # Select top performers
        elite_size = self.population_size // 4
        elite = [self.population[i] for i in sorted_indices[:elite_size]]

        # Generate new population
        new_population = elite.copy()

        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)

            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(self.population[parent1], self.population[parent2])
            else:
                child1, child2 = self.population[parent1].copy(), self.population[parent2].copy()

            # Mutation
            if np.random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.mutation_rate:
                child2 = self._mutate(child2)

            new_population.extend([child1, child2])

        # Update population
        self.population = new_population[:self.population_size]

        return self.population

    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> int:
        """Tournament selection for genetic algorithm."""
        tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return int(winner_idx)

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Randomly swap attributes
        for key in child1.keys():
            if np.random.random() < 0.5:
                child1[key], child2[key] = child2[key], child1[key]

        return child1, child2

    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()

        # Randomly mutate attributes
        for key, values in self.architecture_space.items():
            if np.random.random() < 0.3:  # 30% chance to mutate each attribute
                val = np.random.choice(values)
                if isinstance(values[0], int):
                    mutated[key] = int(val)
                elif isinstance(values[0], float):
                    mutated[key] = float(val)
                else:
                    mutated[key] = str(val)

        return mutated


class AdvancedPiMoESystem(nn.Module):
    """
    Advanced PiMoE system with all enhanced routing capabilities.
    Integrates attention-based routing, hierarchical decisions, dynamic scaling,
    cross-expert communication, and neural architecture search.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_types: List[ExpertType],
        routing_config: AdvancedRoutingConfig,
        enable_nas: bool = False
    ) -> None:
        """
        Initialize AdvancedPiMoESystem.

        Args:
            hidden_size: Hidden size.
            num_experts: Number of experts.
            expert_types: List of expert types.
            routing_config: Advanced routing config.
            enable_nas: Whether to enable NAS.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_types = expert_types
        self.routing_config = routing_config
        self.enable_nas = enable_nas

        # Initialize routing components
        self._initialize_routing_components()

        # Initialize expert networks
        self._initialize_expert_networks()

        # Initialize advanced components
        self._initialize_advanced_components()

    def _initialize_routing_components(self) -> None:
        """Initialize routing components based on strategy."""
        if self.routing_config.strategy == RoutingStrategy.ATTENTION_BASED:
            self.router = AttentionBasedRouter(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                expert_types=self.expert_types,
                attention_heads=self.routing_config.attention_heads
            )
        elif self.routing_config.strategy == RoutingStrategy.HIERARCHICAL:
            self.router = HierarchicalRouter(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                expert_types=self.expert_types,
                hierarchical_levels=self.routing_config.hierarchical_levels
            )
        else:
            # Default to attention-based
            self.router = AttentionBasedRouter(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                expert_types=self.expert_types
            )

    def _initialize_expert_networks(self) -> None:
        """Initialize expert networks."""
        self.experts = nn.ModuleList([
            PiMoEExpert(
                hidden_size=self.hidden_size,
                expert_type=self.expert_types[i % len(self.expert_types)]
            )
            for i in range(self.num_experts)
        ])

    def _initialize_advanced_components(self) -> None:
        """Initialize advanced components."""
        # Dynamic expert scaler
        self.expert_scaler = DynamicExpertScaler(
            base_num_experts=self.num_experts,
            max_num_experts=self.num_experts * 2,
            scaling_threshold=self.routing_config.dynamic_scaling_threshold
        )

        # Cross-expert communicator
        if self.routing_config.cross_expert_communication:
            self.communicator = CrossExpertCommunicator(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts
            )
        else:
            self.communicator = None

        # Neural Architecture Search
        if self.enable_nas:
            self.nas_router = NeuralArchitectureSearchRouter(
                hidden_size=self.hidden_size,
                search_space_size=self.routing_config.nas_search_space
            )
        else:
            self.nas_router = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_advanced_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with advanced routing capabilities.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get routing decisions
        if hasattr(self, 'router'):
            # Check for correct method based on router type
            if return_advanced_info:
                if isinstance(self.router, AttentionBasedRouter):
                     output, routing_info = self.router(
                        hidden_states, attention_mask, return_attention_weights=True
                    )
                elif isinstance(self.router, HierarchicalRouter):
                     output, routing_info = self.router(
                        hidden_states, attention_mask, return_hierarchical_info=True
                     )
                else:
                    output = self.router(hidden_states, attention_mask) # fallback
                    routing_info = None
            else:
                output = self.router(hidden_states, attention_mask)
                routing_info = None
        else:
            output = hidden_states
            routing_info = None

        # Apply expert processing
        # Note: In AttentionBasedRouter, the output IS ALREADY routed and processed.
        # But here logic implies we re-process it?
        # If I look at _apply_expert_processing logic (restored below), it runs ALL experts on the input.
        # If 'output' is already routed hidden states, then running experts on it again is weird.
        # However, preserving logic:
        expert_outputs = self._apply_expert_processing(output, routing_info)

        # Cross-expert communication
        if self.communicator is not None:
            expert_outputs = self.communicator(expert_outputs, list(range(self.num_experts)))

        # Combine expert outputs
        # Note: Dense combination (mean of all experts).
        # This confirms this system is a Dense MoE (or Soft Mixture).
        if expert_outputs:
            final_output = torch.stack(expert_outputs, dim=1).mean(dim=1)
        else:
            final_output = output # Fallback if no experts

        if return_advanced_info:
            advanced_info = {
                'routing_info': routing_info,
                'expert_outputs': expert_outputs, # List of tensors
                'final_output': final_output
            }
            return final_output, advanced_info

        return final_output

    def _apply_expert_processing(
        self,
        hidden_states: torch.Tensor,
        routing_info: Optional[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Apply expert processing based on routing decisions."""
        # Process through each expert
        # Note: This is a Dense operation (all experts run)
        expert_outputs = []
        for expert_id in range(self.num_experts):
            expert_output = self.experts[expert_id](hidden_states)
            expert_outputs.append(expert_output)

        return expert_outputs

    def get_advanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive advanced metrics."""
        metrics = {
            'routing_strategy': self.routing_config.strategy.value,
            'num_experts': self.num_experts,
            'expert_types': [et.value for et in self.expert_types],
            'dynamic_scaling_enabled': self.expert_scaler is not None,
            'cross_expert_communication': self.communicator is not None,
            'neural_architecture_search': self.nas_router is not None
        }

        # Add router-specific metrics
        if hasattr(self.router, 'get_expert_usage_stats'):
            metrics['router_stats'] = self.router.get_expert_usage_stats()

        return metrics


def create_advanced_pimoe_system(
    hidden_size: int,
    num_experts: int = 8,
    expert_types: Optional[List[ExpertType]] = None,
    routing_strategy: RoutingStrategy = RoutingStrategy.ATTENTION_BASED,
    enable_nas: bool = False,
    **kwargs: Any
) -> AdvancedPiMoESystem:
    """
    Factory function to create an advanced PiMoE system.

    Args:
        hidden_size: Hidden layer size.
        num_experts: Number of experts.
        expert_types: List of ExpertTypes.
        routing_strategy: Strategy enum.
        enable_nas: Enable NAS.
        **kwargs: Additional args for Routing Config.

    Returns:
        AdvancedPiMoESystem instance.
    """
    if expert_types is None:
        expert_types = [
            ExpertType.REASONING,
            ExpertType.COMPUTATION,
            ExpertType.MATHEMATICAL,
            ExpertType.LOGICAL,
            ExpertType.LANGUAGE,
            ExpertType.CREATIVE,
            ExpertType.ANALYTICAL
        ]

    # Create routing configuration
    routing_config = AdvancedRoutingConfig(
        strategy=routing_strategy,
        **kwargs
    )

    return AdvancedPiMoESystem(
        hidden_size=hidden_size,
        num_experts=num_experts,
        expert_types=expert_types,
        routing_config=routing_config,
        enable_nas=enable_nas
    )

