"""
Test suite for hybrid optimization core functionality.
"""

import torch
import torch.nn as nn
from optimization_core.hybrid_optimization_core import (
    HybridOptimizationCore, HybridOptimizationConfig, CandidateSelector,
    HybridOptimizationStrategy, create_hybrid_optimization_core
)

class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.norm = nn.LayerNorm(256)
        self.linear2 = nn.Linear(256, 64)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x

def test_hybrid_optimization_config():
    """Test hybrid optimization configuration."""
    config = HybridOptimizationConfig()
    
    assert config.enable_candidate_selection == True
    assert config.enable_tournament_selection == True
    assert config.num_candidates == 5
    assert config.tournament_size == 3
    assert len(config.optimization_strategies) == 4

def test_candidate_selector():
    """Test candidate selection algorithms."""
    config = HybridOptimizationConfig()
    selector = CandidateSelector(config)
    
    candidates = [
        {'strategy': 'A', 'speed_improvement': 1.2, 'memory_efficiency': 1.1, 'accuracy_preservation': 0.99},
        {'strategy': 'B', 'speed_improvement': 1.5, 'memory_efficiency': 2.0, 'accuracy_preservation': 0.97},
        {'strategy': 'C', 'speed_improvement': 1.1, 'memory_efficiency': 1.8, 'accuracy_preservation': 1.0}
    ]
    
    fitness_scores = [selector.evaluate_candidate_fitness(c) for c in candidates]
    
    selected = selector.tournament_selection(candidates, fitness_scores)
    assert selected in candidates
    
    selected = selector.roulette_selection(candidates, fitness_scores)
    assert selected in candidates
    
    selected = selector.rank_selection(candidates, fitness_scores)
    assert selected in candidates

def test_hybrid_optimization_core():
    """Test main hybrid optimization functionality with RL enhancements."""
    config = HybridOptimizationConfig(
        optimization_strategies=['kernel_fusion', 'quantization'],
        enable_rl_optimization=False  # Disable for faster testing
    )
    
    hybrid_core = HybridOptimizationCore(config)
    model = SimpleTestModel()
    
    optimized_model, result = hybrid_core.hybrid_optimize_module(model)
    
    assert optimized_model is not None
    assert isinstance(result, dict)
    assert 'selected_strategy' in result
    assert 'performance_metrics' in result
    
    report = hybrid_core.get_optimization_report()
    assert 'total_optimizations' in report
    assert report['total_optimizations'] == 1

def test_rl_enhanced_optimization():
    """Test RL-enhanced optimization with DAPO, VAPO, and ORZ."""
    config = HybridOptimizationConfig(
        optimization_strategies=['kernel_fusion', 'quantization'],
        enable_rl_optimization=True,
        enable_dapo=True,
        enable_vapo=True,
        enable_orz=True,
        rl_max_episodes=5  # Reduced for testing
    )
    
    hybrid_core = HybridOptimizationCore(config)
    model = SimpleTestModel()
    
    optimized_model, result = hybrid_core.hybrid_optimize_module(model)
    
    assert optimized_model is not None
    assert isinstance(result, dict)
    assert 'selected_strategy' in result
    
    report = hybrid_core.get_optimization_report()
    assert 'rl_optimization_enabled' in report
    assert report['rl_optimization_enabled'] == True
    assert 'dapo_enabled' in report
    assert 'vapo_enabled' in report
    assert 'orz_enabled' in report

def test_factory_function():
    """Test factory function for creating hybrid optimization core."""
    hybrid_core = create_hybrid_optimization_core()
    assert isinstance(hybrid_core, HybridOptimizationCore)
    
    custom_config = {
        'enable_candidate_selection': True,
        'num_candidates': 3,
        'optimization_strategies': ['kernel_fusion', 'quantization']
    }
    
    hybrid_core = create_hybrid_optimization_core(custom_config)
    assert hybrid_core.config.num_candidates == 3
    assert len(hybrid_core.config.optimization_strategies) == 2

if __name__ == "__main__":
    test_hybrid_optimization_config()
    test_candidate_selector()
    test_hybrid_optimization_core()
    test_rl_enhanced_optimization()
    test_factory_function()
    print("✅ All hybrid optimization tests with RL enhancements passed!")
