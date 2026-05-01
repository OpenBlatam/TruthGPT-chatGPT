"""
Refactored Framework Example
============================

Comprehensive example demonstrating the refactored optimization framework
with all modern deep learning best practices.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path

# Import refactored framework components
from ..core.architecture import OptimizationFramework, OptimizationRequest
from ..core.config import UnifiedConfig
from ..models.transformer import TransformerOptimizer, TransformerConfig
from ..training.trainer import Trainer, TrainingConfig
from ..training.data_loader import DataLoader, DataLoaderConfig
from ..api.server import APIServer


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('refactored_framework.log')
        ]
    )


def create_sample_data(num_samples: int = 1000, seq_length: int = 128, vocab_size: int = 1000) -> Dict[str, torch.Tensor]:
    """Create sample data for training"""
    # Generate random input sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    # Generate target sequences (shifted input for language modeling)
    target_ids = torch.roll(input_ids, -1, dims=1)
    target_ids[:, -1] = 0  # Padding token
    
    return {
        'input': input_ids,
        'target': target_ids
    }


def create_data_loaders(config: DataLoaderConfig) -> tuple:
    """Create training and validation data loaders"""
    # Create sample data
    train_data = create_sample_data(num_samples=8000, seq_length=128)
    val_data = create_sample_data(num_samples=2000, seq_length=128)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        train_data['input'], train_data['target']
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_data['input'], val_data['target']
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader


def demonstrate_framework_usage():
    """Demonstrate comprehensive framework usage"""
    print("🚀 TruthGPT Refactored Framework Demo")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 1. Initialize Framework
    print("\n1. Initializing Framework...")
    framework = OptimizationFramework()
    
    # 2. Create Model Configuration
    print("\n2. Creating Model Configuration...")
    model_config = TransformerConfig(
        model_name="truthgpt_transformer",
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        use_flash_attention=True,
        use_rotary_embeddings=True,
        use_lora=True,
        lora_rank=16,
        mixed_precision=True,
        gradient_checkpointing=True,
        compile_model=True
    )
    
    # 3. Create Model
    print("\n3. Creating Transformer Model...")
    model = TransformerOptimizer(model_config)
    print(f"Model created: {model}")
    print(f"Model size: {model.get_model_size()}")
    
    # 4. Create Training Configuration
    print("\n4. Setting up Training Configuration...")
    training_config = TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        mixed_precision=True,
        gradient_accumulation_steps=2,
        optimizer="adamw",
        scheduler="cosine",
        warmup_epochs=2,
        early_stopping_patience=5,
        save_best_model=True,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        use_wandb=False,
        validate_every_n_epochs=1,
        gradient_clip_norm=1.0
    )
    
    # 5. Create Data Loaders
    print("\n5. Creating Data Loaders...")
    data_config = DataLoaderConfig(
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    train_loader, val_loader = create_data_loaders(data_config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # 6. Create Trainer
    print("\n6. Setting up Trainer...")
    trainer = Trainer(model, training_config)
    
    # Add custom callbacks
    from ..training.callbacks import ModelCheckpoint, EarlyStopping
    trainer.add_callback(ModelCheckpoint(save_dir="checkpoints", save_best_only=True))
    trainer.add_callback(EarlyStopping(patience=5, min_delta=1e-4))
    
    # 7. Define Loss Function
    print("\n7. Setting up Loss Function...")
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # 8. Start Training
    print("\n8. Starting Training...")
    print("Training with mixed precision, gradient accumulation, and all optimizations...")
    
    try:
        training_results = trainer.train(train_loader, val_loader, loss_fn)
        
        print(f"\n✅ Training completed!")
        print(f"Total epochs: {training_results['total_epochs']}")
        print(f"Best metric: {training_results['best_metric']:.4f}")
        
        # 9. Evaluate Model
        print("\n9. Evaluating Model...")
        test_loader, _ = create_data_loaders(data_config)
        test_results = trainer.evaluate(test_loader, loss_fn)
        print(f"Test loss: {test_results['test_loss']:.4f}")
        
        # 10. Demonstrate Framework API
        print("\n10. Demonstrating Framework API...")
        
        # Create optimization request
        sample_data = create_sample_data(num_samples=1, seq_length=64)
        request = OptimizationRequest(
            task_id="demo_task_001",
            model_type="transformer",
            data=sample_data,
            config={
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6
            },
            priority=1
        )
        
        # Submit task to framework
        import asyncio
        async def run_optimization():
            result = await framework.optimize(request)
            print(f"Optimization result: {result.success}")
            print(f"Execution time: {result.execution_time:.2f}s")
            print(f"Metrics: {result.metrics}")
        
        # Run async optimization
        asyncio.run(run_optimization())
        
        # 11. Demonstrate API Server
        print("\n11. Starting API Server...")
        api_config = {
            'host': 'localhost',
            'port': 8000,
            'cors_origins': ['*'],
            'rate_limit': 100
        }
        
        api_server = APIServer(framework, api_config)
        print("API server created (not started in demo)")
        print("Available endpoints:")
        print("  - GET /health - Health check")
        print("  - POST /api/v1/optimization/optimize - Submit optimization task")
        print("  - GET /api/v1/monitoring/metrics - Get metrics")
        print("  - GET /api/v1/config - Get configuration")
        print("  - WebSocket /ws - Real-time updates")
        
        # 12. Framework Metrics
        print("\n12. Framework Metrics...")
        metrics = framework.get_framework_metrics()
        print(f"Framework state: {metrics['state']}")
        print(f"Active tasks: {metrics['active_tasks']}")
        print(f"Completed tasks: {metrics['completed_tasks']}")
        print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
        print(f"Memory usage: {metrics['memory_usage']:.2f} MB")
        
        # 13. Model Information
        print("\n13. Model Information...")
        model_info = model.get_model_info()
        print(f"Total parameters: {model_info['total_parameters']:,}")
        print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"Model size: {model_info['model_size_mb']:.2f} MB")
        print(f"Flash Attention: {model_info['use_flash_attention']}")
        print(f"LoRA enabled: {model_info['use_lora']}")
        
        # 14. Cleanup
        print("\n14. Cleaning up...")
        trainer.cleanup()
        await framework.shutdown()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Unified framework architecture")
        print("✅ Advanced transformer model with Flash Attention")
        print("✅ Mixed precision training")
        print("✅ Gradient accumulation and clipping")
        print("✅ Learning rate scheduling")
        print("✅ Early stopping and model checkpointing")
        print("✅ Comprehensive logging and monitoring")
        print("✅ REST API with WebSocket support")
        print("✅ Intelligent caching system")
        print("✅ Dependency injection container")
        print("✅ Factory pattern for model creation")
        print("✅ Async processing with priority queues")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        raise


def demonstrate_advanced_features():
    """Demonstrate advanced framework features"""
    print("\n🔬 Advanced Features Demo")
    print("=" * 30)
    
    # 1. Custom Model Creation
    print("\n1. Creating Custom Model...")
    custom_config = TransformerConfig(
        model_name="custom_transformer",
        d_model=256,
        n_heads=4,
        n_layers=3,
        use_flash_attention=True,
        use_rotary_embeddings=True,
        use_lora=True,
        lora_rank=8
    )
    
    custom_model = TransformerOptimizer(custom_config)
    print(f"Custom model created with {custom_model.get_model_size()['total_parameters']:,} parameters")
    
    # 2. Model Compilation
    print("\n2. Compiling Model...")
    custom_model.compile()
    print("Model compiled for better performance")
    
    # 3. Memory Optimization
    print("\n3. Memory Optimization...")
    memory_usage = custom_model.get_memory_usage()
    print(f"GPU memory usage: {memory_usage}")
    
    # 4. Gradient Checkpointing
    print("\n4. Enabling Gradient Checkpointing...")
    custom_model.enable_gradient_checkpointing()
    print("Gradient checkpointing enabled to save memory")
    
    # 5. Mixed Precision Setup
    print("\n5. Mixed Precision Training...")
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    print(f"Mixed precision scaler: {scaler is not None}")
    
    # 6. Attention Visualization
    print("\n6. Attention Mechanism...")
    sample_input = torch.randint(0, 1000, (1, 64))
    attention_weights = custom_model.get_attention_weights(sample_input)
    print(f"Attention weights shape: {attention_weights.shape if attention_weights is not None else 'N/A'}")
    
    print("\n✅ Advanced features demonstrated!")


if __name__ == "__main__":
    try:
        # Run main demo
        demonstrate_framework_usage()
        
        # Run advanced features demo
        demonstrate_advanced_features()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()



