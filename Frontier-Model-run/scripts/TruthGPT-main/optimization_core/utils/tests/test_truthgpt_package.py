"""
TruthGPT Enhanced Utils Test Suite
Comprehensive test suite for all TruthGPT enhanced utilities
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all TruthGPT utilities
from truthgpt_enhanced_utils import (
    TruthGPTEnhancedConfig,
    TruthGPTPerformanceProfiler,
    TruthGPTAdvancedOptimizer,
    TruthGPTEnhancedManager
)

from truthgpt_advanced_training import (
    TruthGPTTrainingConfig,
    TruthGPTAdvancedTrainer
)

from truthgpt_advanced_evaluation import (
    TruthGPTEvaluationConfig,
    TruthGPTAdvancedEvaluator
)

# Import package functions
from __init__ import (
    quick_truthgpt_setup,
    quick_truthgpt_training,
    quick_truthgpt_evaluation,
    complete_truthgpt_workflow,
    truthgpt_optimization_context,
    truthgpt_training_context,
    truthgpt_evaluation_context
)


class TestTruthGPTQuickStart:
    """Tests for quick start functions."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
    
    @pytest.fixture
    def dummy_dataloader(self):
        """Create dummy data loader."""
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 100, (20, 10))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)
    
    def test_quick_truthgpt_setup(self, simple_model):
        """Test quick TruthGPT setup."""
        optimized_model, manager = quick_truthgpt_setup(
            simple_model, 
            optimization_level="advanced", 
            precision="fp16"
        )
        
        assert optimized_model is not None
        assert manager is not None
        assert isinstance(optimized_model, nn.Module)
        assert isinstance(manager, TruthGPTEnhancedManager)
    
    def test_quick_truthgpt_training(self, simple_model, dummy_dataloader):
        """Test quick TruthGPT training."""
        trained_model = quick_truthgpt_training(
            simple_model,
            dummy_dataloader,
            learning_rate=1e-4,
            max_epochs=1,
            mixed_precision=False
        )
        
        assert trained_model is not None
        assert isinstance(trained_model, nn.Module)
    
    def test_quick_truthgpt_evaluation(self, simple_model, dummy_dataloader):
        """Test quick TruthGPT evaluation."""
        device = torch.device("cpu")
        
        metrics = quick_truthgpt_evaluation(
            simple_model,
            dummy_dataloader,
            device,
            "language_modeling"
        )
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics
    
    def test_complete_truthgpt_workflow(self, simple_model, dummy_dataloader):
        """Test complete TruthGPT workflow."""
        result = complete_truthgpt_workflow(
            simple_model,
            dummy_dataloader,
            dummy_dataloader,
            optimization_level="advanced",
            training_epochs=1
        )
        
        assert isinstance(result, dict)
        assert 'optimized_model' in result
        assert 'trained_model' in result
        assert 'evaluation_metrics' in result
        assert 'enhanced_metrics' in result
        assert 'manager' in result
        
        assert result['optimized_model'] is not None
        assert result['trained_model'] is not None
        assert isinstance(result['evaluation_metrics'], dict)
        assert isinstance(result['enhanced_metrics'], dict)
        assert result['manager'] is not None


class TestTruthGPTContextManagers:
    """Tests for context managers."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
    
    @pytest.fixture
    def dummy_dataloader(self):
        """Create dummy data loader."""
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 100, (20, 10))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)
    
    def test_truthgpt_optimization_context(self, simple_model):
        """Test TruthGPT optimization context manager."""
        with truthgpt_optimization_context(simple_model, "advanced", "fp16") as (optimized_model, manager):
            assert optimized_model is not None
            assert manager is not None
            assert isinstance(optimized_model, nn.Module)
            assert isinstance(manager, TruthGPTEnhancedManager)
    
    def test_truthgpt_training_context(self, simple_model, dummy_dataloader):
        """Test TruthGPT training context manager."""
        with truthgpt_training_context(
            simple_model, 
            dummy_dataloader, 
            max_epochs=1, 
            mixed_precision=False
        ) as trained_model:
            assert trained_model is not None
            assert isinstance(trained_model, nn.Module)
    
    def test_truthgpt_evaluation_context(self, simple_model, dummy_dataloader):
        """Test TruthGPT evaluation context manager."""
        device = torch.device("cpu")
        
        with truthgpt_evaluation_context(
            simple_model, 
            dummy_dataloader, 
            device, 
            "language_modeling"
        ) as metrics:
            assert isinstance(metrics, dict)
            assert 'loss' in metrics
            assert 'perplexity' in metrics
            assert 'accuracy' in metrics


class TestTruthGPTPackageIntegration:
    """Tests for package integration."""
    
    def test_package_imports(self):
        """Test that all package imports work correctly."""
        # Test enhanced utils imports
        from truthgpt_enhanced_utils import (
            TruthGPTEnhancedConfig,
            TruthGPTPerformanceProfiler,
            TruthGPTAdvancedOptimizer,
            TruthGPTEnhancedManager
        )
        
        # Test advanced training imports
        from truthgpt_advanced_training import (
            TruthGPTTrainingConfig,
            TruthGPTAdvancedTrainer
        )
        
        # Test advanced evaluation imports
        from truthgpt_advanced_evaluation import (
            TruthGPTEvaluationConfig,
            TruthGPTAdvancedEvaluator
        )
        
        # All imports should work without errors
        assert True
    
    def test_package_version(self):
        """Test package version information."""
        from __init__ import __version__, __author__, __description__
        
        assert __version__ == "2.0.0"
        assert __author__ == "TruthGPT Optimization Core Team"
        assert "TruthGPT" in __description__
    
    def test_package_exports(self):
        """Test package exports."""
        from __init__ import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 50  # Should have many exports
        
        # Check for key exports
        key_exports = [
            'TruthGPTEnhancedConfig',
            'TruthGPTAdvancedTrainer',
            'TruthGPTAdvancedEvaluator',
            'quick_truthgpt_setup',
            'quick_truthgpt_training',
            'quick_truthgpt_evaluation',
            'complete_truthgpt_workflow'
        ]
        
        for export in key_exports:
            assert export in __all__


class TestTruthGPTErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_optimization_level(self):
        """Test handling of invalid optimization level."""
        model = nn.Linear(10, 10)
        
        # Should handle invalid optimization level gracefully
        try:
            optimized_model, manager = quick_truthgpt_setup(
                model, 
                optimization_level="invalid_level"
            )
            assert optimized_model is not None
        except Exception as e:
            # Should handle error gracefully
            assert "invalid" in str(e).lower() or "error" in str(e).lower()
    
    def test_invalid_precision(self):
        """Test handling of invalid precision."""
        model = nn.Linear(10, 10)
        
        # Should handle invalid precision gracefully
        try:
            optimized_model, manager = quick_truthgpt_setup(
                model, 
                precision="invalid_precision"
            )
            assert optimized_model is not None
        except Exception as e:
            # Should handle error gracefully
            assert "invalid" in str(e).lower() or "error" in str(e).lower()
    
    def test_empty_model(self):
        """Test handling of empty model."""
        # Should handle empty model gracefully
        try:
            optimized_model, manager = quick_truthgpt_setup(None)
            assert optimized_model is None
            assert manager is not None
        except Exception as e:
            # Should handle error gracefully
            assert "none" in str(e).lower() or "error" in str(e).lower()


class TestTruthGPTPerformance:
    """Performance tests for TruthGPT utilities."""
    
    def test_optimization_performance(self):
        """Test optimization performance."""
        import time
        
        model = nn.Sequential(
            nn.Embedding(1000, 100),
            nn.Linear(100, 1000)
        )
        
        start_time = time.time()
        
        optimized_model, manager = quick_truthgpt_setup(
            model, 
            optimization_level="advanced"
        )
        
        optimization_time = time.time() - start_time
        
        assert optimized_model is not None
        assert optimization_time < 5.0  # Should complete within 5 seconds
    
    def test_training_performance(self):
        """Test training performance."""
        import time
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        # Create minimal data
        input_ids = torch.randint(0, 100, (8, 10))
        labels = torch.randint(0, 100, (8, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        start_time = time.time()
        
        trained_model = quick_truthgpt_training(
            model,
            dataloader,
            max_epochs=1,
            mixed_precision=False
        )
        
        training_time = time.time() - start_time
        
        assert trained_model is not None
        assert training_time < 10.0  # Should complete within 10 seconds
    
    def test_evaluation_performance(self):
        """Test evaluation performance."""
        import time
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        # Create minimal data
        input_ids = torch.randint(0, 100, (8, 10))
        labels = torch.randint(0, 100, (8, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        
        start_time = time.time()
        
        metrics = quick_truthgpt_evaluation(
            model,
            dataloader,
            device,
            "language_modeling"
        )
        
        evaluation_time = time.time() - start_time
        
        assert isinstance(metrics, dict)
        assert evaluation_time < 5.0  # Should complete within 5 seconds


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

