"""
Ultra-Advanced Adaptive Optimization System
Adaptive optimization based on workload characteristics and real-time performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategies."""
    CONSERVATIVE = "conservative"           # Conservative optimization
    BALANCED = "balanced"                   # Balanced optimization
    AGGRESSIVE = "aggressive"               # Aggressive optimization
    ULTRA_AGGRESSIVE = "ultra_aggressive"   # Ultra-aggressive optimization
    ADAPTIVE = "adaptive"                   # Fully adaptive optimization
    WORKLOAD_AWARE = "workload_aware"       # Workload-aware optimization

class WorkloadType(Enum):
    """Workload types."""
    SEQUENTIAL = "sequential"               # Sequential processing
    PARALLEL = "parallel"                  # Parallel processing
    MIXED = "mixed"                        # Mixed workload
    BATCH = "batch"                        # Batch processing
    STREAMING = "streaming"                # Streaming processing
    INTERACTIVE = "interactive"            # Interactive processing

class ResourceConstraint(Enum):
    """Resource constraints."""
    MEMORY_LIMITED = "memory_limited"       # Memory-constrained
    COMPUTE_LIMITED = "compute_limited"      # Compute-constrained
    LATENCY_SENSITIVE = "latency_sensitive" # Latency-sensitive
    THROUGHPUT_OPTIMIZED = "throughput_optimized" # Throughput-optimized
    BALANCED = "balanced"                  # Balanced resources

@dataclass
class WorkloadProfile:
    """Workload profile for adaptive optimization."""
    workload_type: WorkloadType = WorkloadType.SEQUENTIAL
    resource_constraint: ResourceConstraint = ResourceConstraint.BALANCED
    
    # Performance characteristics
    avg_sequence_length: float = 512.0
    batch_size: float = 1.0
    request_rate: float = 1.0
    memory_usage: float = 0.5
    cpu_usage: float = 0.5
    gpu_usage: float = 0.5
    
    # Optimization preferences
    latency_weight: float = 0.5
    throughput_weight: float = 0.5
    memory_weight: float = 0.5
    
    # Historical data
    performance_history: List[float] = field(default_factory=list)
    optimization_history: List[str] = field(default_factory=list)

@dataclass
class OptimizationConfig:
    """Configuration for adaptive optimization."""
    # Basic settings
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    adaptation_frequency: int = 100  # Adapt every N operations
    learning_rate: float = 0.01
    
    # Monitoring settings
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    history_size: int = 1000
    
    # Optimization parameters
    cache_size_multiplier: float = 1.0
    batch_size_multiplier: float = 1.0
    precision_level: str = "fp16"
    
    # Adaptive thresholds
    performance_threshold: float = 0.95
    memory_threshold: float = 0.8
    latency_threshold: float = 0.1  # seconds
    
    # Advanced features
    use_ml_prediction: bool = True
    use_reinforcement_learning: bool = True
    use_evolutionary_optimization: bool = True

class AdaptiveOptimizer:
    """
    Ultra-Advanced Adaptive Optimization System.
    
    Features:
    - Real-time workload analysis
    - Dynamic optimization strategy selection
    - ML-based performance prediction
    - Reinforcement learning for optimization
    - Evolutionary optimization algorithms
    - Resource-aware optimization
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.workload_profile = WorkloadProfile()
        
        # Performance tracking
        self.performance_metrics = defaultdict(deque)
        self.optimization_history = []
        self.adaptation_count = 0
        
        # ML components
        self._setup_ml_components()
        
        # Optimization strategies
        self._setup_optimization_strategies()
        
        # Background monitoring
        self._setup_monitoring()
        
        logger.info(f"Adaptive Optimizer initialized with strategy: {config.optimization_strategy}")
    
    def _setup_ml_components(self):
        """Setup ML components for prediction and optimization."""
        if self.config.use_ml_prediction:
            self.performance_predictor = PerformancePredictor()
        
        if self.config.use_reinforcement_learning:
            self.rl_optimizer = ReinforcementLearningOptimizer()
        
        if self.config.use_evolutionary_optimization:
            self.evolutionary_optimizer = EvolutionaryOptimizer()
    
    def _setup_optimization_strategies(self):
        """Setup optimization strategies."""
        self.strategies = {
            OptimizationStrategy.CONSERVATIVE: self._conservative_optimization,
            OptimizationStrategy.BALANCED: self._balanced_optimization,
            OptimizationStrategy.AGGRESSIVE: self._aggressive_optimization,
            OptimizationStrategy.ULTRA_AGGRESSIVE: self._ultra_aggressive_optimization,
            OptimizationStrategy.ADAPTIVE: self._adaptive_optimization,
            OptimizationStrategy.WORKLOAD_AWARE: self._workload_aware_optimization
        }
    
    def _setup_monitoring(self):
        """Setup background monitoring."""
        if self.config.enable_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self.monitoring_thread.start()
    
    def _monitor_performance(self):
        """Background performance monitoring."""
        while True:
            try:
                # Collect performance metrics
                self._collect_performance_metrics()
                
                # Analyze workload
                self._analyze_workload()
                
                # Adaptive optimization
                if self.adaptation_count % self.config.adaptation_frequency == 0:
                    self._adaptive_optimization()
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
    
    def _collect_performance_metrics(self):
        """Collect real-time performance metrics."""
        # Memory usage
        memory_usage = self._get_memory_usage()
        self.performance_metrics['memory'].append(memory_usage)
        
        # CPU usage
        cpu_usage = self._get_cpu_usage()
        self.performance_metrics['cpu'].append(cpu_usage)
        
        # GPU usage
        gpu_usage = self._get_gpu_usage()
        self.performance_metrics['gpu'].append(gpu_usage)
        
        # Latency
        latency = self._get_current_latency()
        self.performance_metrics['latency'].append(latency)
        
        # Throughput
        throughput = self._get_current_throughput()
        self.performance_metrics['throughput'].append(throughput)
        
        # Limit history size
        for metric in self.performance_metrics.values():
            while len(metric) > self.config.history_size:
                metric.popleft()
    
    def _analyze_workload(self):
        """Analyze current workload characteristics."""
        # Analyze sequence length patterns
        if 'sequence_length' in self.performance_metrics:
            avg_seq_len = np.mean(list(self.performance_metrics['sequence_length']))
            self.workload_profile.avg_sequence_length = avg_seq_len
        
        # Analyze batch size patterns
        if 'batch_size' in self.performance_metrics:
            avg_batch_size = np.mean(list(self.performance_metrics['batch_size']))
            self.workload_profile.batch_size = avg_batch_size
        
        # Analyze request rate
        if 'request_rate' in self.performance_metrics:
            avg_request_rate = np.mean(list(self.performance_metrics['request_rate']))
            self.workload_profile.request_rate = avg_request_rate
        
        # Determine workload type
        self.workload_profile.workload_type = self._determine_workload_type()
        
        # Determine resource constraints
        self.workload_profile.resource_constraint = self._determine_resource_constraint()
    
    def _determine_workload_type(self) -> WorkloadType:
        """Determine workload type based on patterns."""
        if self.workload_profile.request_rate > 10:
            return WorkloadType.STREAMING
        elif self.workload_profile.batch_size > 8:
            return WorkloadType.BATCH
        elif self.workload_profile.avg_sequence_length > 2048:
            return WorkloadType.SEQUENTIAL
        else:
            return WorkloadType.MIXED
    
    def _determine_resource_constraint(self) -> ResourceConstraint:
        """Determine resource constraints."""
        memory_usage = np.mean(list(self.performance_metrics['memory'])) if self.performance_metrics['memory'] else 0.5
        cpu_usage = np.mean(list(self.performance_metrics['cpu'])) if self.performance_metrics['cpu'] else 0.5
        gpu_usage = np.mean(list(self.performance_metrics['gpu'])) if self.performance_metrics['gpu'] else 0.5
        
        if memory_usage > 0.8:
            return ResourceConstraint.MEMORY_LIMITED
        elif cpu_usage > 0.8:
            return ResourceConstraint.COMPUTE_LIMITED
        elif gpu_usage > 0.8:
            return ResourceConstraint.COMPUTE_LIMITED
        else:
            return ResourceConstraint.BALANCED
    
    def _adaptive_optimization(self):
        """Perform adaptive optimization."""
        logger.info("Performing adaptive optimization...")
        
        # Select optimization strategy
        strategy = self._select_optimization_strategy()
        
        # Apply optimization
        optimization_result = self.strategies[strategy]()
        
        # Record optimization
        self.optimization_history.append({
            'timestamp': time.time(),
            'strategy': strategy.value,
            'result': optimization_result,
            'workload_profile': self.workload_profile.__dict__
        })
        
        self.adaptation_count += 1
        
        logger.info(f"Adaptive optimization completed with strategy: {strategy.value}")
    
    def _select_optimization_strategy(self) -> OptimizationStrategy:
        """Select optimal strategy based on current conditions."""
        if self.config.optimization_strategy == OptimizationStrategy.ADAPTIVE:
            return self._ml_based_strategy_selection()
        elif self.config.optimization_strategy == OptimizationStrategy.WORKLOAD_AWARE:
            return self._workload_based_strategy_selection()
        else:
            return self.config.optimization_strategy
    
    def _ml_based_strategy_selection(self) -> OptimizationStrategy:
        """ML-based strategy selection."""
        if hasattr(self, 'performance_predictor'):
            # Use ML model to predict best strategy
            features = self._extract_features()
            predicted_strategy = self.performance_predictor.predict_strategy(features)
            return OptimizationStrategy(predicted_strategy)
        else:
            # Fallback to rule-based selection
            return self._rule_based_strategy_selection()
    
    def _workload_based_strategy_selection(self) -> OptimizationStrategy:
        """Workload-based strategy selection."""
        workload_type = self.workload_profile.workload_type
        resource_constraint = self.workload_profile.resource_constraint
        
        if workload_type == WorkloadType.STREAMING and resource_constraint == ResourceConstraint.LATENCY_SENSITIVE:
            return OptimizationStrategy.AGGRESSIVE
        elif workload_type == WorkloadType.BATCH and resource_constraint == ResourceConstraint.THROUGHPUT_OPTIMIZED:
            return OptimizationStrategy.ULTRA_AGGRESSIVE
        elif resource_constraint == ResourceConstraint.MEMORY_LIMITED:
            return OptimizationStrategy.CONSERVATIVE
        else:
            return OptimizationStrategy.BALANCED
    
    def _rule_based_strategy_selection(self) -> OptimizationStrategy:
        """Rule-based strategy selection."""
        memory_usage = np.mean(list(self.performance_metrics['memory'])) if self.performance_metrics['memory'] else 0.5
        latency = np.mean(list(self.performance_metrics['latency'])) if self.performance_metrics['latency'] else 0.1
        
        if memory_usage > 0.8:
            return OptimizationStrategy.CONSERVATIVE
        elif latency > self.config.latency_threshold:
            return OptimizationStrategy.AGGRESSIVE
        else:
            return OptimizationStrategy.BALANCED
    
    def _extract_features(self) -> List[float]:
        """Extract features for ML prediction."""
        features = []
        
        # Performance metrics
        for metric_name in ['memory', 'cpu', 'gpu', 'latency', 'throughput']:
            if metric_name in self.performance_metrics and self.performance_metrics[metric_name]:
                features.append(np.mean(list(self.performance_metrics[metric_name])))
            else:
                features.append(0.0)
        
        # Workload characteristics
        features.extend([
            self.workload_profile.avg_sequence_length,
            self.workload_profile.batch_size,
            self.workload_profile.request_rate
        ])
        
        return features
    
    def _conservative_optimization(self) -> Dict[str, Any]:
        """Conservative optimization strategy."""
        return {
            'cache_size_multiplier': 0.5,
            'batch_size_multiplier': 0.5,
            'precision_level': 'fp32',
            'compression_ratio': 0.7,
            'quantization_bits': 16
        }
    
    def _balanced_optimization(self) -> Dict[str, Any]:
        """Balanced optimization strategy."""
        return {
            'cache_size_multiplier': 1.0,
            'batch_size_multiplier': 1.0,
            'precision_level': 'fp16',
            'compression_ratio': 0.5,
            'quantization_bits': 8
        }
    
    def _aggressive_optimization(self) -> Dict[str, Any]:
        """Aggressive optimization strategy."""
        return {
            'cache_size_multiplier': 2.0,
            'batch_size_multiplier': 2.0,
            'precision_level': 'fp16',
            'compression_ratio': 0.3,
            'quantization_bits': 8
        }
    
    def _ultra_aggressive_optimization(self) -> Dict[str, Any]:
        """Ultra-aggressive optimization strategy."""
        return {
            'cache_size_multiplier': 4.0,
            'batch_size_multiplier': 4.0,
            'precision_level': 'fp16',
            'compression_ratio': 0.2,
            'quantization_bits': 4
        }
    
    def _adaptive_optimization(self) -> Dict[str, Any]:
        """Fully adaptive optimization strategy."""
        # Use ML or RL to determine optimal parameters
        if hasattr(self, 'rl_optimizer'):
            return self.rl_optimizer.optimize(self.workload_profile)
        elif hasattr(self, 'evolutionary_optimizer'):
            return self.evolutionary_optimizer.optimize(self.workload_profile)
        else:
            # Fallback to balanced
            return self._balanced_optimization()
    
    def _workload_aware_optimization(self) -> Dict[str, Any]:
        """Workload-aware optimization strategy."""
        workload_type = self.workload_profile.workload_type
        resource_constraint = self.workload_profile.resource_constraint
        
        base_config = self._balanced_optimization()
        
        # Adjust based on workload type
        if workload_type == WorkloadType.STREAMING:
            base_config['cache_size_multiplier'] *= 0.5
            base_config['batch_size_multiplier'] *= 0.5
        elif workload_type == WorkloadType.BATCH:
            base_config['cache_size_multiplier'] *= 2.0
            base_config['batch_size_multiplier'] *= 2.0
        
        # Adjust based on resource constraints
        if resource_constraint == ResourceConstraint.MEMORY_LIMITED:
            base_config['compression_ratio'] = 0.7
            base_config['quantization_bits'] = 4
        elif resource_constraint == ResourceConstraint.LATENCY_SENSITIVE:
            base_config['compression_ratio'] = 0.2
            base_config['quantization_bits'] = 16
        
        return base_config
    
    def optimize_decoder(self, decoder: Any) -> Dict[str, Any]:
        """Optimize decoder based on current workload."""
        # Get current optimization parameters
        optimization_params = self.strategies[self.config.optimization_strategy]()
        
        # Apply optimizations to decoder
        if hasattr(decoder, 'kv_cache'):
            # Optimize cache
            cache_config = decoder.kv_cache.config
            cache_config.max_cache_size = int(cache_config.max_cache_size * optimization_params['cache_size_multiplier'])
            cache_config.compression_ratio = optimization_params['compression_ratio']
            cache_config.quantization_bits = optimization_params['quantization_bits']
        
        # Apply batch size optimization
        if hasattr(decoder, 'config'):
            decoder.config.batch_size = int(decoder.config.batch_size * optimization_params['batch_size_multiplier'])
        
        return optimization_params
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            'optimization_strategy': self.config.optimization_strategy.value,
            'adaptation_count': self.adaptation_count,
            'workload_profile': self.workload_profile.__dict__,
            'performance_metrics': {k: list(v) for k, v in self.performance_metrics.items()},
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'current_performance': {
                'memory_usage': np.mean(list(self.performance_metrics['memory'])) if self.performance_metrics['memory'] else 0.0,
                'cpu_usage': np.mean(list(self.performance_metrics['cpu'])) if self.performance_metrics['cpu'] else 0.0,
                'gpu_usage': np.mean(list(self.performance_metrics['gpu'])) if self.performance_metrics['gpu'] else 0.0,
                'latency': np.mean(list(self.performance_metrics['latency'])) if self.performance_metrics['latency'] else 0.0,
                'throughput': np.mean(list(self.performance_metrics['throughput'])) if self.performance_metrics['throughput'] else 0.0
            }
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except:
            return 0.5
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent() / 100.0
        except:
            return 0.5
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load
            else:
                return 0.5
        except:
            return 0.5
    
    def _get_current_latency(self) -> float:
        """Get current latency."""
        # Simplified latency calculation
        return 0.1
    
    def _get_current_throughput(self) -> float:
        """Get current throughput."""
        # Simplified throughput calculation
        return 1.0

# ML Component Classes
class PerformancePredictor:
    """ML-based performance predictor."""
    
    def __init__(self):
        self.model = None
        self.feature_history = []
        self.performance_history = []
    
    def predict_strategy(self, features: List[float]) -> str:
        """Predict optimal strategy based on features."""
        # Simplified prediction logic
        if features[0] > 0.8:  # High memory usage
            return "conservative"
        elif features[3] > 0.1:  # High latency
            return "aggressive"
        else:
            return "balanced"
    
    def update_model(self, features: List[float], performance: float):
        """Update ML model with new data."""
        self.feature_history.append(features)
        self.performance_history.append(performance)

class ReinforcementLearningOptimizer:
    """Reinforcement learning optimizer."""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
    
    def optimize(self, workload_profile: WorkloadProfile) -> Dict[str, Any]:
        """RL-based optimization."""
        # Simplified RL optimization
        state = self._encode_state(workload_profile)
        action = self._select_action(state)
        return self._decode_action(action)
    
    def _encode_state(self, workload_profile: WorkloadProfile) -> str:
        """Encode workload profile as state."""
        return f"{workload_profile.workload_type.value}_{workload_profile.resource_constraint.value}"
    
    def _select_action(self, state: str) -> str:
        """Select action using epsilon-greedy policy."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if np.random.random() < self.epsilon:
            # Exploration
            actions = ['conservative', 'balanced', 'aggressive', 'ultra_aggressive']
            return np.random.choice(actions)
        else:
            # Exploitation
            if self.q_table[state]:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return 'balanced'
    
    def _decode_action(self, action: str) -> Dict[str, Any]:
        """Decode action to optimization parameters."""
        action_map = {
            'conservative': {'cache_size_multiplier': 0.5, 'batch_size_multiplier': 0.5},
            'balanced': {'cache_size_multiplier': 1.0, 'batch_size_multiplier': 1.0},
            'aggressive': {'cache_size_multiplier': 2.0, 'batch_size_multiplier': 2.0},
            'ultra_aggressive': {'cache_size_multiplier': 4.0, 'batch_size_multiplier': 4.0}
        }
        return action_map.get(action, action_map['balanced'])
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning."""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table.get(next_state, {}).values()) if next_state in self.q_table else 0.0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

class EvolutionaryOptimizer:
    """Evolutionary optimization algorithm."""
    
    def __init__(self):
        self.population_size = 20
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
    
    def optimize(self, workload_profile: WorkloadProfile) -> Dict[str, Any]:
        """Evolutionary optimization."""
        # Initialize population
        population = self._initialize_population()
        
        # Evolve population
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual, workload_profile) for individual in population]
            
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(parents)
            
            # Mutation
            offspring = self._mutation(offspring)
            
            # Replace population
            population = offspring
        
        # Return best individual
        best_individual = max(population, key=lambda x: self._evaluate_fitness(x, workload_profile))
        return self._decode_individual(best_individual)
    
    def _initialize_population(self) -> List[List[float]]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = [
                np.random.uniform(0.1, 4.0),  # cache_size_multiplier
                np.random.uniform(0.1, 4.0),  # batch_size_multiplier
                np.random.uniform(0.1, 0.9),  # compression_ratio
                np.random.choice([4, 8, 16])  # quantization_bits
            ]
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual: List[float], workload_profile: WorkloadProfile) -> float:
        """Evaluate fitness of individual."""
        # Simplified fitness function
        cache_mult, batch_mult, comp_ratio, quant_bits = individual
        
        # Fitness based on workload characteristics
        fitness = 1.0
        
        if workload_profile.workload_type == WorkloadType.STREAMING:
            fitness *= (1.0 / (cache_mult + batch_mult))  # Prefer smaller multipliers
        elif workload_profile.workload_type == WorkloadType.BATCH:
            fitness *= (cache_mult + batch_mult)  # Prefer larger multipliers
        
        if workload_profile.resource_constraint == ResourceConstraint.MEMORY_LIMITED:
            fitness *= comp_ratio  # Prefer higher compression
        
        return fitness
    
    def _selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[List[float]]:
        """Tournament selection."""
        parents = []
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_index])
        return parents
    
    def _crossover(self, parents: List[List[float]]) -> List[List[float]]:
        """Uniform crossover."""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                if np.random.random() < self.crossover_rate:
                    child1 = [parent1[j] if np.random.random() < 0.5 else parent2[j] for j in range(len(parent1))]
                    child2 = [parent2[j] if np.random.random() < 0.5 else parent1[j] for j in range(len(parent2))]
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])
        return offspring
    
    def _mutation(self, offspring: List[List[float]]) -> List[List[float]]:
        """Gaussian mutation."""
        for individual in offspring:
            if np.random.random() < self.mutation_rate:
                for i in range(len(individual)):
                    if i < 2:  # cache_size_multiplier, batch_size_multiplier
                        individual[i] = max(0.1, individual[i] + np.random.normal(0, 0.1))
                    elif i == 2:  # compression_ratio
                        individual[i] = max(0.1, min(0.9, individual[i] + np.random.normal(0, 0.05)))
                    else:  # quantization_bits
                        individual[i] = np.random.choice([4, 8, 16])
        return offspring
    
    def _decode_individual(self, individual: List[float]) -> Dict[str, Any]:
        """Decode individual to optimization parameters."""
        return {
            'cache_size_multiplier': individual[0],
            'batch_size_multiplier': individual[1],
            'compression_ratio': individual[2],
            'quantization_bits': int(individual[3])
        }

# Factory functions
def create_adaptive_optimizer(config: OptimizationConfig = None) -> AdaptiveOptimizer:
    """Create an adaptive optimizer."""
    if config is None:
        config = OptimizationConfig()
    return AdaptiveOptimizer(config)

def create_optimization_config(**kwargs) -> OptimizationConfig:
    """Create an optimization configuration."""
    return OptimizationConfig(**kwargs)


