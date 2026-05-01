"""
TruthGPT Complete Integration Tests
End-to-end tests for all TruthGPT utilities working together
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from truthgpt_enhanced_utils import (
    TruthGPTEnhancedConfig,
    TruthGPTEnhancedManager,
    create_enhanced_truthgpt_manager,
    quick_enhanced_truthgpt_optimization
)

from truthgpt_advanced_training import (
    TruthGPTTrainingConfig,
    TruthGPTAdvancedTrainer,
    create_advanced_trainer,
    quick_advanced_training
)

from truthgpt_advanced_evaluation import (
    TruthGPTEvaluationConfig,
    TruthGPTAdvancedEvaluator,
    create_advanced_evaluator,
    quick_evaluation
)


class TestTruthGPTCompleteIntegration:
    """Complete integration tests for TruthGPT utilities."""
    
    @pytest.fixture
    def truthgpt_model(self):
        """Create a comprehensive TruthGPT model."""
        class TruthGPTModel(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=128, num_layers=3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=4,
                        dim_feedforward=hidden_size * 4,
                        dropout=0.1
                    ),
                    num_layers=num_layers
                )
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.dropout(x)
                x = self.transformer(x)
                x = self.lm_head(x)
                return x
        
        return TruthGPTModel()
    
    @pytest.fixture
    def training_data(self):
        """Create training data."""
        input_ids = torch.randint(0, 1000, (100, 20))
        labels = torch.randint(0, 1000, (100, 20))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pytest.fixture
    def validation_data(self):
        """Create validation data."""
        input_ids = torch.randint(0, 1000, (40, 20))
        labels = torch.randint(0, 1000, (40, 20))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=8, shuffle=False)
    
    def test_complete_workflow(self, truthgpt_model, training_data, validation_data):
        """Test complete TruthGPT workflow: optimization -> training -> evaluation."""
        
        # Step 1: Enhanced Optimization
        print("🚀 Step 1: Enhanced Optimization")
        enhanced_config = TruthGPTEnhancedConfig(
            optimization_level="ultra",
            precision="fp16",
            enable_quantization=True,
            enable_pruning=True,
            enable_memory_optimization=True,
            enable_monitoring=True
        )
        
        enhanced_manager = create_enhanced_truthgpt_manager(enhanced_config)
        optimized_model = enhanced_manager.optimize_model_enhanced(truthgpt_model, "balanced")
        
        assert optimized_model is not None
        assert isinstance(optimized_model, nn.Module)
        
        # Step 2: Advanced Training
        print("📚 Step 2: Advanced Training")
        training_config = TruthGPTTrainingConfig(
            learning_rate=1e-4,
            max_epochs=2,
            batch_size=8,
            mixed_precision=False,  # Disable for testing
            gradient_checkpointing=True,
            tensorboard_logging=False,
            wandb_logging=False,
            early_stopping=False
        )
        
        trainer = create_advanced_trainer(training_config)
        trained_model = trainer.train(optimized_model, training_data, validation_data)
        
        assert trained_model is not None
        assert isinstance(trained_model, nn.Module)
        
        # Step 3: Advanced Evaluation
        print("📊 Step 3: Advanced Evaluation")
        evaluation_config = TruthGPTEvaluationConfig(
            compute_accuracy=True,
            compute_perplexity=True,
            compute_diversity=True,
            compute_coherence=True,
            save_reports=False,
            create_visualizations=False
        )
        
        evaluator = create_advanced_evaluator(evaluation_config)
        device = torch.device("cpu")
        
        evaluation_metrics = evaluator.evaluate_model(
            trained_model, validation_data, device, "language_modeling"
        )
        
        assert isinstance(evaluation_metrics, dict)
        assert 'loss' in evaluation_metrics
        assert 'perplexity' in evaluation_metrics
        assert 'accuracy' in evaluation_metrics
        assert 'diversity_score' in evaluation_metrics
        assert 'coherence_score' in evaluation_metrics
        
        # Step 4: Get comprehensive metrics
        print("📈 Step 4: Comprehensive Metrics")
        enhanced_metrics = enhanced_manager.get_enhanced_metrics()
        evaluation_summary = evaluator.get_evaluation_summary()
        
        assert isinstance(enhanced_metrics, dict)
        assert isinstance(evaluation_summary, dict)
        
        print("✅ Complete TruthGPT workflow successful!")
        print(f"   - Optimization: {len(enhanced_metrics['optimization_stats']['optimizers_available'])} optimizers")
        print(f"   - Training: {len(trainer.training_metrics)} metric types")
        print(f"   - Evaluation: {evaluation_summary['total_evaluations']} evaluations")
    
    def test_quick_workflow(self, truthgpt_model, training_data, validation_data):
        """Test quick workflow functions."""
        
        # Quick optimization
        optimized_model = quick_enhanced_truthgpt_optimization(
            truthgpt_model,
            optimization_level="advanced",
            precision="fp16",
            device="cpu"
        )
        
        assert optimized_model is not None
        
        # Quick training
        trained_model = quick_advanced_training(
            optimized_model,
            training_data,
            validation_data,
            learning_rate=1e-4,
            max_epochs=1,
            mixed_precision=False
        )
        
        assert trained_model is not None
        
        # Quick evaluation
        device = torch.device("cpu")
        metrics = quick_evaluation(trained_model, validation_data, device, "language_modeling")
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics
    
    def test_model_comparison_workflow(self, training_data, validation_data):
        """Test model comparison workflow."""
        
        # Create multiple models
        model1 = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.Linear(64, 1000)
        )
        
        model2 = nn.Sequential(
            nn.Embedding(1000, 64),
            nn.Linear(64, 1000)
        )
        
        # Optimize both models
        optimized_model1 = quick_enhanced_truthgpt_optimization(model1, "balanced")
        optimized_model2 = quick_enhanced_truthgpt_optimization(model2, "aggressive")
        
        # Train both models
        trained_model1 = quick_advanced_training(optimized_model1, training_data, validation_data, max_epochs=1)
        trained_model2 = quick_advanced_training(optimized_model2, training_data, validation_data, max_epochs=1)
        
        # Compare models
        evaluation_config = TruthGPTEvaluationConfig(
            compare_models=True,
            save_reports=False,
            create_visualizations=False
        )
        
        evaluator = create_advanced_evaluator(evaluation_config)
        device = torch.device("cpu")
        
        models = {
            'model1': trained_model1,
            'model2': trained_model2
        }
        
        comparison_results = evaluator.compare_models(models, validation_data, device)
        
        assert isinstance(comparison_results, dict)
        assert 'models' in comparison_results
        assert 'best_model' in comparison_results
        assert 'best_metric' in comparison_results
        assert 'model1' in comparison_results['models']
        assert 'model2' in comparison_results['models']
    
    def test_error_recovery_workflow(self, truthgpt_model, training_data):
        """Test error recovery in complete workflow."""
        
        # Create configuration with error recovery
        enhanced_config = TruthGPTEnhancedConfig(
            enable_error_recovery=True,
            max_retries=3,
            enable_fault_tolerance=True
        )
        
        enhanced_manager = create_enhanced_truthgpt_manager(enhanced_config)
        
        # This should handle any errors gracefully
        try:
            optimized_model = enhanced_manager.optimize_model_enhanced(truthgpt_model, "balanced")
            assert optimized_model is not None
            
            # Check error recovery stats
            metrics = enhanced_manager.get_enhanced_metrics()
            assert 'error_recovery_stats' in metrics
            
        except Exception as e:
            # Should handle errors gracefully
            assert enhanced_manager.error_recovery['retry_count'] <= enhanced_config.max_retries
    
    def test_caching_workflow(self, truthgpt_model):
        """Test caching in complete workflow."""
        
        # Create configuration with caching
        enhanced_config = TruthGPTEnhancedConfig(enable_caching=True)
        enhanced_manager = create_enhanced_truthgpt_manager(enhanced_config)
        
        # First optimization (cache miss)
        optimized_model1 = enhanced_manager.optimize_model_enhanced(truthgpt_model, "balanced")
        initial_misses = enhanced_manager.caching_system['cache_misses']
        
        # Second optimization (cache hit)
        optimized_model2 = enhanced_manager.optimize_model_enhanced(truthgpt_model, "balanced")
        final_hits = enhanced_manager.caching_system['cache_hits']
        
        assert optimized_model1 is not None
        assert optimized_model2 is not None
        
        # Should have cache hit
        assert final_hits > 0 or initial_misses == enhanced_manager.caching_system['cache_misses']
    
    def test_monitoring_workflow(self, truthgpt_model, training_data):
        """Test monitoring in complete workflow."""
        
        # Create configuration with monitoring
        enhanced_config = TruthGPTEnhancedConfig(enable_monitoring=True)
        enhanced_manager = create_enhanced_truthgpt_manager(enhanced_config)
        
        # Optimize model
        optimized_model = enhanced_manager.optimize_model_enhanced(truthgpt_model, "balanced")
        
        # Check monitoring
        metrics = enhanced_manager.get_enhanced_metrics()
        assert 'monitoring_stats' in metrics
        assert isinstance(metrics['monitoring_stats'], dict)
        
        # Training with monitoring
        training_config = TruthGPTTrainingConfig(
            max_epochs=1,
            tensorboard_logging=False,
            wandb_logging=False
        )
        
        trainer = create_advanced_trainer(training_config)
        trained_model = trainer.train(optimized_model, training_data)
        
        assert trained_model is not None
        assert len(trainer.training_metrics) > 0


class TestTruthGPTPerformanceIntegration:
    """Performance integration tests."""
    
    def test_performance_optimization_workflow(self):
        """Test performance optimization workflow."""
        
        # Create large model
        large_model = nn.Sequential(
            nn.Embedding(10000, 512),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048),
                num_layers=6
            ),
            nn.Linear(512, 10000)
        )
        
        # Performance optimization
        optimized_model = quick_enhanced_truthgpt_optimization(
            large_model,
            optimization_level="ultra",
            precision="fp16",
            device="cpu"
        )
        
        assert optimized_model is not None
        
        # Check model size reduction
        original_params = sum(p.numel() for p in large_model.parameters())
        optimized_params = sum(p.numel() for p in optimized_model.parameters())
        
        # Should have same or fewer parameters (due to pruning)
        assert optimized_params <= original_params
    
    def test_memory_efficient_workflow(self):
        """Test memory efficient workflow."""
        
        # Create model with gradient checkpointing
        model = nn.Sequential(
            nn.Embedding(1000, 256),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024),
                num_layers=4
            ),
            nn.Linear(256, 1000)
        )
        
        # Memory optimization
        enhanced_config = TruthGPTEnhancedConfig(
            enable_memory_optimization=True,
            enable_gradient_checkpointing=True
        )
        
        enhanced_manager = create_enhanced_truthgpt_manager(enhanced_config)
        optimized_model = enhanced_manager.optimize_model_enhanced(model, "balanced")
        
        assert optimized_model is not None
        
        # Check memory optimization
        metrics = enhanced_manager.get_enhanced_metrics()
        assert 'optimization_stats' in metrics


class TestTruthGPTEdgeCaseIntegration:
    """Edge case integration tests."""
    
    def test_minimal_workflow(self):
        """Test minimal workflow with smallest possible components."""
        
        # Minimal model
        minimal_model = nn.Linear(10, 10)
        
        # Minimal data
        input_ids = torch.randint(0, 10, (4, 5))
        labels = torch.randint(0, 10, (4, 5))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Minimal optimization
        optimized_model = quick_enhanced_truthgpt_optimization(minimal_model, "conservative")
        
        # Minimal training
        trained_model = quick_advanced_training(
            optimized_model, dataloader, max_epochs=1, mixed_precision=False
        )
        
        # Minimal evaluation
        device = torch.device("cpu")
        metrics = quick_evaluation(trained_model, dataloader, device)
        
        assert optimized_model is not None
        assert trained_model is not None
        assert isinstance(metrics, dict)
    
    def test_empty_data_workflow(self):
        """Test workflow with empty data."""
        
        model = nn.Linear(10, 10)
        
        # Empty data loader
        empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
        empty_dataloader = DataLoader(empty_dataset, batch_size=1)
        
        # Should handle gracefully
        try:
            optimized_model = quick_enhanced_truthgpt_optimization(model, "conservative")
            assert optimized_model is not None
            
            # Training should handle empty data
            trained_model = quick_advanced_training(
                optimized_model, empty_dataloader, max_epochs=1, mixed_precision=False
            )
            assert trained_model is not None
            
        except Exception as e:
            # Should handle error gracefully
            assert "empty" in str(e).lower() or "no data" in str(e).lower()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

