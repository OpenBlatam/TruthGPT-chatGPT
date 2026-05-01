"""
TruthGPT Advanced Training Tests
Comprehensive tests for TruthGPT advanced training utilities
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

from truthgpt_advanced_training import (
    TruthGPTTrainingConfig,
    TruthGPTAdvancedTrainer,
    create_advanced_trainer,
    quick_advanced_training,
    advanced_training_context
)


class TestTruthGPTTrainingConfig:
    """Tests for TruthGPTTrainingConfig."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TruthGPTTrainingConfig()
        
        assert config.model_name == "truthgpt"
        assert config.model_size == "base"
        assert config.architecture == "transformer"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.batch_size == 32
        assert config.max_epochs == 100
        assert config.mixed_precision is True
        assert config.gradient_checkpointing is True
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = TruthGPTTrainingConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model_name' in config_dict
        assert 'learning_rate' in config_dict
        assert 'mixed_precision' in config_dict
        assert config_dict['model_name'] == "truthgpt"
    
    def test_custom_config(self):
        """Test custom training configuration."""
        config = TruthGPTTrainingConfig(
            model_name="custom_truthgpt",
            model_size="large",
            learning_rate=2e-4,
            batch_size=64,
            max_epochs=50,
            mixed_precision=False
        )
        
        assert config.model_name == "custom_truthgpt"
        assert config.model_size == "large"
        assert config.learning_rate == 2e-4
        assert config.batch_size == 64
        assert config.max_epochs == 50
        assert config.mixed_precision is False


class TestTruthGPTAdvancedTrainer:
    """Tests for TruthGPTAdvancedTrainer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TruthGPTTrainingConfig(
            max_epochs=2,
            batch_size=4,
            mixed_precision=False,  # Disable for testing
            gradient_checkpointing=False,
            data_parallel=False,
            tensorboard_logging=False,
            wandb_logging=False
        )
    
    @pytest.fixture
    def trainer(self, config):
        """Create advanced trainer."""
        return TruthGPTAdvancedTrainer(config)
    
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
        # Create dummy data
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 100, (20, 10))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)
    
    def test_trainer_initialization(self, trainer, config):
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.device is not None
        assert trainer.training_metrics is not None
        assert trainer.validation_metrics is not None
    
    def test_setup_device(self, trainer):
        """Test device setup."""
        device = trainer._setup_device()
        assert isinstance(device, torch.device)
    
    def test_setup_model(self, trainer, simple_model):
        """Test model setup."""
        device = torch.device("cpu")  # Use CPU for testing
        trainer.device = device
        
        setup_model = trainer.setup_model(simple_model)
        
        assert setup_model is not None
        assert isinstance(setup_model, nn.Module)
    
    def test_setup_optimizer(self, trainer, simple_model):
        """Test optimizer setup."""
        optimizer = trainer.setup_optimizer(simple_model)
        
        assert optimizer is not None
        assert isinstance(optimizer, optim.Optimizer)
    
    def test_setup_scheduler(self, trainer, simple_model):
        """Test scheduler setup."""
        optimizer = trainer.setup_optimizer(simple_model)
        scheduler = trainer.setup_scheduler(optimizer, 100)
        
        assert scheduler is not None
        assert isinstance(scheduler, optim.lr_scheduler._LRScheduler)
    
    def test_compute_loss(self, trainer, simple_model):
        """Test loss computation."""
        device = torch.device("cpu")
        trainer.device = device
        
        # Create dummy batch
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))
        
        loss = trainer._compute_loss(simple_model, (input_ids, labels))
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_evaluate(self, trainer, simple_model, dummy_dataloader):
        """Test model evaluation."""
        device = torch.device("cpu")
        trainer.device = device
        
        eval_metrics = trainer.evaluate(simple_model, dummy_dataloader)
        
        assert isinstance(eval_metrics, dict)
        assert 'loss' in eval_metrics
        assert 'perplexity' in eval_metrics
        assert 'accuracy' in eval_metrics
    
    def test_checkpoint_save_load(self, trainer, simple_model):
        """Test checkpoint saving and loading."""
        device = torch.device("cpu")
        trainer.device = device
        
        # Setup model and optimizer
        model = trainer.setup_model(simple_model)
        optimizer = trainer.setup_optimizer(model)
        trainer.setup_scheduler(optimizer, 100)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.config.save_dir = temp_dir
            
            # Save checkpoint
            trainer._save_checkpoint(model, 0, {'loss': 0.5})
            
            # Check if checkpoint file exists
            checkpoint_dir = Path(temp_dir)
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            assert len(checkpoint_files) > 0
            
            # Load checkpoint
            checkpoint_path = checkpoint_files[0]
            start_epoch = trainer._load_checkpoint(model, str(checkpoint_path))
            
            assert start_epoch == 1  # Should be epoch + 1


class TestTruthGPTTrainingIntegration:
    """Integration tests for TruthGPT training."""
    
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
    def training_dataloader(self):
        """Create training data loader."""
        input_ids = torch.randint(0, 100, (40, 10))
        labels = torch.randint(0, 100, (40, 10))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pytest.fixture
    def validation_dataloader(self):
        """Create validation data loader."""
        input_ids = torch.randint(0, 100, (20, 10))
        labels = torch.randint(0, 100, (20, 10))
        
        dataset = TensorDataset(input_ids, labels)
        return DataLoader(dataset, batch_size=8, shuffle=False)
    
    def test_quick_training(self, truthgpt_model, training_dataloader):
        """Test quick training function."""
        trained_model = quick_advanced_training(
            truthgpt_model,
            training_dataloader,
            learning_rate=1e-3,
            max_epochs=1,
            mixed_precision=False
        )
        
        assert trained_model is not None
        assert isinstance(trained_model, nn.Module)
    
    def test_training_context_manager(self, truthgpt_model, training_dataloader):
        """Test training context manager."""
        config = TruthGPTTrainingConfig(
            max_epochs=1,
            mixed_precision=False,
            tensorboard_logging=False,
            wandb_logging=False
        )
        
        with advanced_training_context(truthgpt_model, config) as trainer:
            assert trainer is not None
            assert isinstance(trainer, TruthGPTAdvancedTrainer)
    
    def test_full_training_workflow(self, truthgpt_model, training_dataloader, validation_dataloader):
        """Test complete training workflow."""
        config = TruthGPTTrainingConfig(
            max_epochs=1,
            batch_size=8,
            mixed_precision=False,
            gradient_checkpointing=False,
            data_parallel=False,
            tensorboard_logging=False,
            wandb_logging=False,
            early_stopping=False
        )
        
        trainer = TruthGPTAdvancedTrainer(config)
        
        # Train model
        trained_model = trainer.train(truthgpt_model, training_dataloader, validation_dataloader)
        
        assert trained_model is not None
        assert isinstance(trained_model, nn.Module)
        
        # Check training metrics
        assert len(trainer.training_metrics) > 0
    
    def test_early_stopping(self, truthgpt_model, training_dataloader, validation_dataloader):
        """Test early stopping mechanism."""
        config = TruthGPTTrainingConfig(
            max_epochs=5,
            batch_size=8,
            mixed_precision=False,
            early_stopping=True,
            early_stopping_patience=1,
            early_stopping_metric="loss",
            early_stopping_mode="min",
            tensorboard_logging=False,
            wandb_logging=False
        )
        
        trainer = TruthGPTAdvancedTrainer(config)
        
        # Train model
        trained_model = trainer.train(truthgpt_model, training_dataloader, validation_dataloader)
        
        assert trained_model is not None
        # Early stopping should prevent full training
        assert trainer.early_stopping_counter >= 0


class TestTruthGPTTrainingEdgeCases:
    """Edge case tests for TruthGPT training."""
    
    def test_empty_dataloader(self):
        """Test handling of empty data loader."""
        config = TruthGPTTrainingConfig(max_epochs=1)
        trainer = TruthGPTAdvancedTrainer(config)
        
        # Create empty data loader
        empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
        empty_dataloader = DataLoader(empty_dataset, batch_size=1)
        
        model = nn.Linear(10, 1)
        
        # Should handle gracefully
        try:
            trained_model = trainer.train(model, empty_dataloader)
            assert trained_model is not None
        except Exception as e:
            # Should handle error gracefully
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    def test_single_batch_training(self):
        """Test training with single batch."""
        config = TruthGPTTrainingConfig(max_epochs=1, batch_size=4)
        trainer = TruthGPTAdvancedTrainer(config)
        
        # Create single batch data loader
        input_ids = torch.randint(0, 100, (4, 10))
        labels = torch.randint(0, 100, (4, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        trained_model = trainer.train(model, dataloader)
        
        assert trained_model is not None
        assert isinstance(trained_model, nn.Module)
    
    def test_large_model_training(self):
        """Test training with larger model."""
        config = TruthGPTTrainingConfig(
            max_epochs=1,
            batch_size=2,
            mixed_precision=False,
            gradient_checkpointing=True
        )
        trainer = TruthGPTAdvancedTrainer(config)
        
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
        input_ids = torch.randint(0, 1000, (8, 20))
        labels = torch.randint(0, 1000, (8, 20))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=2)
        
        trained_model = trainer.train(model, dataloader)
        
        assert trained_model is not None
        assert isinstance(trained_model, nn.Module)


# Performance tests
class TestTruthGPTTrainingPerformance:
    """Performance tests for TruthGPT training."""
    
    def test_training_speed(self):
        """Test training speed."""
        config = TruthGPTTrainingConfig(
            max_epochs=1,
            batch_size=8,
            mixed_precision=False,
            tensorboard_logging=False,
            wandb_logging=False
        )
        trainer = TruthGPTAdvancedTrainer(config)
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        # Create data loader
        input_ids = torch.randint(0, 100, (32, 10))
        labels = torch.randint(0, 100, (32, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=8)
        
        import time
        start_time = time.time()
        
        trained_model = trainer.train(model, dataloader)
        
        training_time = time.time() - start_time
        
        assert trained_model is not None
        assert training_time < 10.0  # Should complete within 10 seconds
    
    def test_memory_usage(self):
        """Test memory usage during training."""
        config = TruthGPTTrainingConfig(
            max_epochs=1,
            batch_size=4,
            mixed_precision=False,
            gradient_checkpointing=True
        )
        trainer = TruthGPTAdvancedTrainer(config)
        
        model = nn.Sequential(
            nn.Embedding(100, 50),
            nn.Linear(50, 100)
        )
        
        # Create data loader
        input_ids = torch.randint(0, 100, (16, 10))
        labels = torch.randint(0, 100, (16, 10))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        trained_model = trainer.train(model, dataloader)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert trained_model is not None
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

