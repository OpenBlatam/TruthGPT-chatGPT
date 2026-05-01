"""
TruthGPT Advanced Evaluation Tests
Comprehensive tests for TruthGPT advanced evaluation utilities
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

from truthgpt_advanced_evaluation import (
    TruthGPTEvaluationConfig,
    TruthGPTAdvancedEvaluator,
    create_advanced_evaluator,
    quick_evaluation
)


class TestTruthGPTEvaluationConfig:
    """Tests for TruthGPTEvaluationConfig."""
    
    def test_default_config(self):
        """Test default evaluation configuration."""
        config = TruthGPTEvaluationConfig()
        
        assert config.compute_accuracy is True
        assert config.compute_perplexity is True
        assert config.compute_bleu is True
        assert config.compute_rouge is True
        assert config.compute_diversity is True
        assert config.compute_coherence is True
        assert config.max_generation_length == 512
        assert config.num_generations == 10
        assert config.temperature == 0.7
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = TruthGPTEvaluationConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'compute_accuracy' in config_dict
        assert 'compute_perplexity' in config_dict
        assert 'max_generation_length' in config_dict
        assert config_dict['compute_accuracy'] is True
    
    def test_custom_config(self):
        """Test custom evaluation configuration."""
        config = TruthGPTEvaluationConfig(
            compute_accuracy=False,
            compute_perplexity=False,
            compute_bleu=False,
            max_generation_length=256,
            num_generations=5,
            temperature=0.5
        )
        
        assert config.compute_accuracy is False
        assert config.compute_perplexity is False
        assert config.compute_bleu is False
        assert config.max_generation_length == 256
        assert config.num_generations == 5
        assert config.temperature == 0.5


class TestTruthGPTAdvancedEvaluator:
    """Tests for TruthGPTAdvancedEvaluator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TruthGPTEvaluationConfig(
            compute_accuracy=True,
            compute_perplexity=True,
            compute_bleu=False,  # Disable for testing
            compute_rouge=False,
            save_reports=False,
            create_visualizations=False
        )
    
    @pytest.fixture
    def evaluator(self, config):
        """Create advanced evaluator."""
        return TruthGPTAdvancedEvaluator(config)
    
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
        from torch.utils.data import DataLoader, TensorDataset
        
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 100, (20, 10))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)
    
    def test_evaluator_initialization(self, evaluator, config):
        """Test evaluator initialization."""
        assert evaluator.config == config
        assert evaluator.metrics is not None
        assert evaluator.detailed_metrics is not None
        assert evaluator.comparison_results is not None
    
    def test_evaluate_language_modeling(self, evaluator, simple_model, dummy_dataloader):
        """Test language modeling evaluation."""
        device = torch.device("cpu")
        
        metrics = evaluator._evaluate_language_modeling(simple_model, dummy_dataloader, device)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics
        assert 'tokens' in metrics
        assert metrics['loss'] >= 0
        assert metrics['perplexity'] >= 0
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_classification(self, evaluator, simple_model, dummy_dataloader):
        """Test classification evaluation."""
        device = torch.device("cpu")
        
        metrics = evaluator._evaluate_classification(simple_model, dummy_dataloader, device)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_evaluate_generation(self, evaluator, simple_model, dummy_dataloader):
        """Test generation evaluation."""
        device = torch.device("cpu")
        
        # Add generate method to model for testing
        class GenerationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 50)
                self.linear = nn.Linear(50, 100)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.linear(x)
                return x
            
            def generate(self, input_ids, max_length=10, temperature=1.0, top_p=1.0, top_k=50):
                # Simple generation for testing
                return torch.randint(0, 100, (input_ids.size(0), max_length))
        
        generation_model = GenerationModel()
        
        metrics = evaluator._evaluate_generation(generation_model, dummy_dataloader, device)
        
        assert isinstance(metrics, dict)
        assert 'num_generated' in metrics
        assert 'avg_length' in metrics
        assert metrics['num_generated'] > 0
        assert metrics['avg_length'] > 0
    
    def test_evaluate_generic(self, evaluator, simple_model, dummy_dataloader):
        """Test generic evaluation."""
        device = torch.device("cpu")
        
        metrics = evaluator._evaluate_generic(simple_model, dummy_dataloader, device)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_compute_diversity_metrics(self, evaluator, simple_model, dummy_dataloader):
        """Test diversity metrics computation."""
        device = torch.device("cpu")
        
        diversity_metrics = evaluator._compute_diversity_metrics(simple_model, dummy_dataloader, device)
        
        assert isinstance(diversity_metrics, dict)
        assert 'unique_ngrams' in diversity_metrics
        assert 'diversity_score' in diversity_metrics
        assert 0 <= diversity_metrics['diversity_score'] <= 1
    
    def test_compute_coherence_metrics(self, evaluator, simple_model, dummy_dataloader):
        """Test coherence metrics computation."""
        device = torch.device("cpu")
        
        coherence_metrics = evaluator._compute_coherence_metrics(simple_model, dummy_dataloader, device)
        
        assert isinstance(coherence_metrics, dict)
        assert 'coherence_score' in coherence_metrics
        assert 0 <= coherence_metrics['coherence_score'] <= 1
    
    def test_compute_relevance_metrics(self, evaluator, simple_model, dummy_dataloader):
        """Test relevance metrics computation."""
        device = torch.device("cpu")
        
        relevance_metrics = evaluator._compute_relevance_metrics(simple_model, dummy_dataloader, device)
        
        assert isinstance(relevance_metrics, dict)
        assert 'relevance_score' in relevance_metrics
        assert 0 <= relevance_metrics['relevance_score'] <= 1
    
    def test_compute_bleu_approximation(self, evaluator):
        """Test BLEU score approximation."""
        generated = ["hello world", "good morning"]
        references = ["hello world", "good morning"]
        
        bleu_score = evaluator._compute_bleu_approximation(generated, references)
        
        assert isinstance(bleu_score, float)
        assert 0 <= bleu_score <= 1
    
    def test_compare_models(self, evaluator, dummy_dataloader):
        """Test model comparison."""
        device = torch.device("cpu")
        
        # Create multiple models
        model1 = nn.Sequential(nn.Embedding(100, 50), nn.Linear(50, 100))
        model2 = nn.Sequential(nn.Embedding(100, 50), nn.Linear(50, 100))
        
        models = {
            'model1': model1,
            'model2': model2
        }
        
        comparison_results = evaluator.compare_models(models, dummy_dataloader, device)
        
        assert isinstance(comparison_results, dict)
        assert 'models' in comparison_results
        assert 'best_model' in comparison_results
        assert 'best_metric' in comparison_results
        assert 'model1' in comparison_results['models']
        assert 'model2' in comparison_results['models']
    
    def test_get_evaluation_summary(self, evaluator):
        """Test getting evaluation summary."""
        summary = evaluator.get_evaluation_summary()
        
        assert isinstance(summary, dict)
        assert 'detailed_metrics' in summary
        assert 'comparison_results' in summary
        assert 'total_evaluations' in summary
        assert 'evaluation_history' in summary


class TestTruthGPTEvaluationIntegration:
    """Integration tests for TruthGPT evaluation."""
    
    @pytest.fixture
    def truthgpt_model(self):
        """Create a TruthGPT-like model."""
        class SimpleTruthGPT(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 50)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=50, nhead=2, dim_feedforward=100),
                    num_layers=2
                )
                self.lm_head = nn.Linear(50, 100)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = self.lm_head(x)
                return x
        
        return SimpleTruthGPT()
    
    @pytest.fixture
    def evaluation_dataloader(self):
        """Create evaluation data loader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 100, (20, 10))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)
    
    def test_quick_evaluation(self, truthgpt_model, evaluation_dataloader):
        """Test quick evaluation function."""
        device = torch.device("cpu")
        
        metrics = quick_evaluation(truthgpt_model, evaluation_dataloader, device, "language_modeling")
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics
    
    def test_comprehensive_evaluation(self, truthgpt_model, evaluation_dataloader):
        """Test comprehensive model evaluation."""
        config = TruthGPTEvaluationConfig(
            compute_accuracy=True,
            compute_perplexity=True,
            compute_diversity=True,
            compute_coherence=True,
            compute_relevance=True,
            save_reports=False,
            create_visualizations=False
        )
        
        evaluator = TruthGPTAdvancedEvaluator(config)
        device = torch.device("cpu")
        
        metrics = evaluator.evaluate_model(truthgpt_model, evaluation_dataloader, device, "language_modeling")
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics
        assert 'diversity_score' in metrics
        assert 'coherence_score' in metrics
        assert 'relevance_score' in metrics
        assert 'evaluation_time' in metrics
    
    def test_multiple_task_evaluation(self, truthgpt_model, evaluation_dataloader):
        """Test evaluation on multiple tasks."""
        config = TruthGPTEvaluationConfig(save_reports=False, create_visualizations=False)
        evaluator = TruthGPTAdvancedEvaluator(config)
        device = torch.device("cpu")
        
        # Test different task types
        task_types = ["language_modeling", "classification", "generation"]
        
        for task_type in task_types:
            metrics = evaluator.evaluate_model(truthgpt_model, evaluation_dataloader, device, task_type)
            
            assert isinstance(metrics, dict)
            assert 'evaluation_time' in metrics
            assert metrics['evaluation_time'] > 0
        
        # Check evaluation history
        summary = evaluator.get_evaluation_summary()
        assert len(summary['evaluation_history']) == len(task_types)


class TestTruthGPTEvaluationEdgeCases:
    """Edge case tests for TruthGPT evaluation."""
    
    def test_empty_dataloader(self):
        """Test handling of empty data loader."""
        config = TruthGPTEvaluationConfig(save_reports=False, create_visualizations=False)
        evaluator = TruthGPTAdvancedEvaluator(config)
        
        # Create empty data loader
        from torch.utils.data import DataLoader, TensorDataset
        empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
        empty_dataloader = DataLoader(empty_dataset, batch_size=1)
        
        model = nn.Linear(10, 1)
        device = torch.device("cpu")
        
        # Should handle gracefully
        try:
            metrics = evaluator.evaluate_model(model, empty_dataloader, device, "language_modeling")
            assert isinstance(metrics, dict)
        except Exception as e:
            # Should handle error gracefully
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    def test_single_batch_evaluation(self):
        """Test evaluation with single batch."""
        config = TruthGPTEvaluationConfig(save_reports=False, create_visualizations=False)
        evaluator = TruthGPTAdvancedEvaluator(config)
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        # Create single batch data loader
        from torch.utils.data import DataLoader, TensorDataset
        input_ids = torch.randint(0, 100, (4, 10))
        labels = torch.randint(0, 100, (4, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        
        metrics = evaluator.evaluate_model(model, dataloader, device, "language_modeling")
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics
    
    def test_large_model_evaluation(self):
        """Test evaluation with larger model."""
        config = TruthGPTEvaluationConfig(save_reports=False, create_visualizations=False)
        evaluator = TruthGPTAdvancedEvaluator(config)
        
        # Create larger model
        model = nn.Sequential(
            nn.Embedding(1000, 200),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=200, nhead=4, dim_feedforward=400),
                num_layers=4
            ),
            nn.Linear(200, 1000)
        )
        
        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        input_ids = torch.randint(0, 1000, (8, 20))
        labels = torch.randint(0, 1000, (8, 20))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=2)
        
        device = torch.device("cpu")
        
        metrics = evaluator.evaluate_model(model, dataloader, device, "language_modeling")
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'accuracy' in metrics


# Performance tests
class TestTruthGPTEvaluationPerformance:
    """Performance tests for TruthGPT evaluation."""
    
    def test_evaluation_speed(self):
        """Test evaluation speed."""
        config = TruthGPTEvaluationConfig(save_reports=False, create_visualizations=False)
        evaluator = TruthGPTAdvancedEvaluator(config)
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        input_ids = torch.randint(0, 100, (32, 10))
        labels = torch.randint(0, 100, (32, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=8)
        
        device = torch.device("cpu")
        
        import time
        start_time = time.time()
        
        metrics = evaluator.evaluate_model(model, dataloader, device, "language_modeling")
        
        evaluation_time = time.time() - start_time
        
        assert isinstance(metrics, dict)
        assert evaluation_time < 5.0  # Should complete within 5 seconds
        assert metrics['evaluation_time'] < 5.0
    
    def test_memory_usage(self):
        """Test memory usage during evaluation."""
        config = TruthGPTEvaluationConfig(save_reports=False, create_visualizations=False)
        evaluator = TruthGPTAdvancedEvaluator(config)
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        input_ids = torch.randint(0, 100, (16, 10))
        labels = torch.randint(0, 100, (16, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        device = torch.device("cpu")
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        metrics = evaluator.evaluate_model(model, dataloader, device, "language_modeling")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert isinstance(metrics, dict)
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

