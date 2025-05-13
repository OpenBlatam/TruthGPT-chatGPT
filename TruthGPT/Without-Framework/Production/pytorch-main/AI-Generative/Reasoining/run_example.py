import torch
from torch.utils.data import TensorDataset, DataLoader
from modular import ExecutionPipeline, ModelConfig
import yaml
import os
from pathlib import Path

def create_sample_data(batch_size: int, seq_len: int, hidden_size: int, num_classes: int):
    """Create sample data for testing."""
    # Create random input data
    x = torch.randn(batch_size, seq_len, hidden_size)
    # Create random target labels
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y

def setup_experiment():
    """Setup experiment directories and logging."""
    # Create necessary directories
    os.makedirs('runs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

def main():
    # Setup experiment
    setup_experiment()
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    batch_size = config['training']['batch_size']
    seq_len = config['data']['max_length']
    hidden_size = config['model']['hidden_size']
    num_classes = config['model']['output_size']
    
    # Create training and validation datasets
    train_x, train_y = create_sample_data(batch_size, seq_len, hidden_size, num_classes)
    val_x, val_y = create_sample_data(batch_size, seq_len, hidden_size, num_classes)
    
    # Create data loaders
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize execution pipeline
    pipeline = ExecutionPipeline(config)
    
    # Start training
    print("Starting training...")
    pipeline.train(train_loader, val_loader)
    print("Training completed!")

if __name__ == "__main__":
    main() 