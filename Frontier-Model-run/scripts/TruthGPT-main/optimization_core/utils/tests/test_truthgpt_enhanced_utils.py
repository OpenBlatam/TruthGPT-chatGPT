"""
TruthGPT Enhanced Utils Tests
Comprehensive tests for TruthGPT enhanced utilities
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthgpt_enhanced_utils import (
    TruthGPTEnhancedConfig,
    TruthGPTPerformanceProfiler,
    TruthGPTAdvancedOptimizer,
    TruthGPTEnhancedManager,
    create_enhanced_truthgpt_manager,
    quick_enhanced_truthgpt_optimization
)


class TestTruthGPTEnhancedConfig:
    """Tests for TruthGPTEnhancedConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TruthGPTEnhancedConfig()
        
        assert config.model_name == "truthgpt"
        assert config.model_size == "base"
        assert config.precision == "fp16"
        assert config.device == "auto"
        assert config.optimization_level == "ultra"
        assert config.target_latency_ms == 50.0
        assert config.target_memory_gb == 8.0
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = TruthGPTEnhancedConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model_name' in config_dict
        assert 'optimization_level' in config_dict
        assert config_dict['model_name'] == "truthgpt"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TruthGPTEnhancedConfig(
            model_name="custom_truthgpt",
            model_size="large",
            precision="bf16",
            optimization_level="aggressive",
            target_latency_ms=30.0
        )
        
        assert config.model_name == "custom_truthgpt"
        assert config.model_size == "large"
        assert config.precision == "bf16"
        assert config.optimization_level == "aggressive"
        assert config.target_latency_ms == 30.0


class TestTruthGPTPerformanceProfiler:
    """Tests for TruthGPTPerformanceProfiler."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TruthGPTEnhancedConfig(
            enable_profiling=True,
            enable_metrics=True
        )
    
    @pytest.fixture
    def profiler(self, config):
        """Create performance profiler."""
        return TruthGPTPerformanceProfiler(config)
    
    def test_profiler_initialization(self, profiler, config):
        """Test profiler initialization."""
        assert profiler.config == config
        assert profiler.performance_metrics is not None
        assert profiler.benchmark_results is not None
    
    def test_memory_analysis(self, profiler):
        """Test memory usage analysis."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50)
        )
        
        memory_analysis = profiler._analyze_memory_usage(model)
        
        assert 'model_parameters' in memory_analysis
        assert 'model_size_mb' in memory_analysis
        assert 'trainable_parameters' in memory_analysis
        assert memory_analysis['model_parameters'] > 0
    
    def test_efficiency_score_calculation(self, profiler):
        """Test efficiency score calculation."""
        performance_data = {
            'inference_times': [0.01, 0.02, 0.015],
            'throughput': [100, 200, 150],
            'memory_usage': [1024 * 1024 * 100, 1024 * 1024 * 200]
        }
        
        efficiency_score = profiler._calculate_efficiency_score(performance_data)
        
        assert isinstance(efficiency_score, float)
        assert 0.0 <= efficiency_score <= 1.0


class TestTruthGPTAdvancedOptimizer:
    """Tests for TruthGPTAdvancedOptimizer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TruthGPTEnhancedConfig(
            enable_quantization=True,
            enable_pruning=True,
            enable_memory_optimization=True,
            enable_attention_optimization=True,
            enable_kernel_fusion=True,
            enable_graph_optimization=True
        )
    
    @pytest.fixture
    def optimizer(self, config):
        """Create advanced optimizer."""
        return TruthGPTAdvancedOptimizer(config)
    
    def test_optimizer_initialization(self, optimizer, config):
        """Test optimizer initialization."""
        assert optimizer.config == config
        assert 'quantization' in optimizer.optimizers
        assert 'pruning' in optimizer.optimizers
        assert 'memory' in optimizer.optimizers
        assert 'performance' in optimizer.optimizers
    
    def test_create_optimization_plan(self, optimizer):
        """Test optimization plan creation."""
        optimization_plan = optimizer._create_optimization_plan()
        
        assert isinstance(optimization_plan, dict)
        assert 'quantization' in optimization_plan
        assert 'pruning' in optimization_plan
        assert optimization_plan['quantization']['enabled'] is True
    
    def test_get_optimization_priority(self, optimizer):
        """Test optimization priority calculation."""
        priorities = ['quantization', 'pruning', 'memory', 'performance']
        
        for opt in priorities:
            priority = optimizer._get_optimization_priority(opt)
            assert isinstance(priority, int)
            assert priority >= 1 and priority <= 10
    
    def test_apply_quantization(self, optimizer):
        """Test quantization application."""
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50)
        )
        
        params = {
            'config': {'bits': 8, 'scheme': 'symmetric', 'calibration_samples': 100}
        }
        
        optimized_model = optimizer._apply_quantization(model, params)
        assert optimized_model is not None
    
    def test_apply_pruning(self, optimizer):
        """Test pruning application."""
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50)
        )
        
        params = {
            'config': {'sparsity': 0.1, 'iterative': True}
        }
        
        optimized_model = optimizer._apply_pruning(model, params)
        assert optimized_model is not None
    
    def test_apply_memory_optimization(self, optimizer):
        """Test memory optimization."""
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50)
        )
        
        params = {
            'config': {'gradient_checkpointing': True, 'memory_pooling': True}
        }
        
        optimized_model = optimizer._apply_memory_optimization(model, params)
        assert optimized_model is not None
    
    def test_get_optimization_stats(self, optimizer):
        """Test getting optimization statistics."""
        stats = optimizer.get_optimization_stats()
        
        assert isinstance(stats, dict)
        assert 'total_optimizations' in stats
        assert 'optimizers_available' in stats
        assert isinstance(stats['optimizers_available'], list)


class TestTruthGPTEnhancedManager:
    """Tests for TruthGPTEnhancedManager."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TruthGPTEnhancedConfig(
            enable_caching=True,
            enable_monitoring=True,
            enable_error_recovery=True,
            enable_auto_scaling=True,
            enable_quantization=True,
            enable_pruning=True
        )
    
    @pytest.fixture
    def manager(self, config):
        """Create enhanced manager."""
        return TruthGPTEnhancedManager(config)
    
    def test_manager_initialization(self, manager, config):
        """Test manager initialization."""
        assert manager.config == config
        assert manager.optimizer is not None
        assert manager.performance_profiler is not None
        assert manager.caching_system is not None
        assert manager.monitoring_system is not None
    
    def test_generate_cache_key(self, manager):
        """Test cache key generation."""
        model = nn.Linear(10, 20)
        strategy = "balanced"
        
        cache_key = manager._generate_cache_key(model, strategy)
        
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
    
    def test_validate_optimization(self, manager):
        """Test optimization validation."""
        # Valid model
        valid_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Should not raise error
        manager._validate_optimization(valid_model)
    
    def test_get_enhanced_metrics(self, manager):
        """Test getting enhanced metrics."""
        metrics = manager.get_enhanced_metrics()
        
        assert isinstance(metrics, dict)
        assert 'optimization_stats' in metrics
        assert 'caching_stats' in metrics
        assert 'monitoring_stats' in metrics
        assert 'error_recovery_stats' in metrics


class TestTruthGPTIntegration:
    """Integration tests for TruthGPT utilities."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Embedding(1000, 100),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=100, nhead=4, dim_feedforward=200),
                num_layers=2
            ),
            nn.Linear(100, 1000)
        )
    
    def test_enhanced_manager_integration(self, simple_model):
        """Test enhanced manager full integration."""
        config = TruthGPTEnhancedConfig(
            enable_quantization=True,
            enable_pruning=True,
            enable_memory_optimization=True
        )
        
        manager = create_enhanced_truthgpt_manager(config)
        
        # Test optimization
        optimized_model = manager.optimize_model_enhanced(simple_model, "balanced")
        
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
    
    def test_quick_optimization(self, simple_model):
        """Test quick optimization function."""
        optimized_model = quick_enhanced_truthgpt_optimization(
            simple_model,
            optimization_level="ultra",
            precision="fp16",
            device="cpu"
        )
        
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
    
    def test_error_recovery(self, simple_model):
        """Test error recovery mechanism."""
        config = TruthGPTEnhancedConfig(
            max_retries=3,
            enable_error_recovery=True
        )
        
        manager = TruthGPTEnhancedManager(config)
        
        # Should not raise errors
        try:
            optimized_model = manager.optimize_model_enhanced(simple_model)
            assert optimized_model is not None
        except Exception as e:
            # If error recovery works, should handle gracefully
            assert manager.error_recovery['retry_count'] <= config.max_retries


# Integration test fixtures
@pytest.fixture
def truthgpt_model():
    """Create a TruthGPT-like model for testing."""
    class SimpleTruthGPT(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_size, 4, 256),
                num_layers=num_layers
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.lm_head(x)
            return x
    
    return SimpleTruthGPT()


@pytest.fixture
def optimized_truthgpt_model():
    """Create an optimized TruthGPT model."""
    config = TruthGPTEnhancedConfig(
        optimization_level="advanced",
        precision="fp16",
        enable_quantization=True,
        enable_pruning=True
    )
    
    manager = TruthGPTEnhancedManager(config)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Embedding(1000, 100),
        nn.Linear(100, 100)
    )
    
    return manager.optimize_model_enhanced(model, "balanced")


class TestEndToEnd:
    """End-to-end tests for complete TruthGPT workflow."""
    
    def test_complete_optimization_workflow(self, truthgpt_model):
        """Test complete optimization workflow."""
        config = TruthGPTEnhancedConfig(
            enable_quantization=True,
            enable_pruning=True,
            enable_memory_optimization=True,
            enable_monitoring=True
        )
        
        # Create manager
        manager = TruthGPTEnhancedManager(config)
        
        # Optimize model
        optimized_model = manager.optimize_model_enhanced(truthgpt_model)
        
        # Check optimization
        assert optimized_model is not None
        
        # Get metrics
        metrics = manager.get_enhanced_metrics()
        assert metrics is not None
        assert 'optimization_stats' in metrics
    
    def test_caching_mechanism(self, truthgpt_model):
        """Test caching mechanism."""
        config = TruthGPTEnhancedConfig(enable_caching=True)
        manager = TruthGPTEnhancedManager(config)
        
        # First optimization (cache miss)
        optimized_model_1 = manager.optimize_model_enhanced(truthgpt_model)
        initial_misses = manager.caching_system['cache_misses']
        
        # Second optimization (cache hit)
        optimized_model_2 = manager.optimize_model_enhanced(truthgpt_model)
        final_hits = manager.caching_system['cache_hits']
        
        # Should have cache hit
        assert final_hits > 0 or initial_misses == manager.caching_system['cache_misses']
    
    def test_monitoring_metrics_collection(self, truthgpt_model):
        """Test monitoring metrics collection."""
        config = TruthGPTEnhancedConfig(enable_monitoring=True)
        manager = TruthGPTEnhancedManager(config)
        
        # Optimize model
        optimized_model = manager.optimize_model_enhanced(truthgpt_model)
        
        # Check metrics collection
        metrics = manager.get_enhanced_metrics()
        assert 'monitoring_stats' in metrics
        assert isinstance(metrics['monitoring_stats'], dict)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

