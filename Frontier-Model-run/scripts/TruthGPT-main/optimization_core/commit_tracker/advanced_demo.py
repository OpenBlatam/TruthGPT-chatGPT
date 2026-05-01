"""
Advanced Commit Tracking System Demo
Demonstrates deep learning enhanced commit tracking with performance analytics
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from commit_tracker import (
    CommitTracker, OptimizationCommit, CommitType, CommitStatus,
    create_commit_tracker, track_optimization_commit
)
from version_manager import (
    VersionManager, VersionType, VersionStatus,
    create_version_manager, create_version
)
from optimization_registry import (
    OptimizationRegistry, OptimizationCategory, RegistryStatus,
    create_optimization_registry, register_optimization
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleModel(nn.Module):
    """Sample model for demonstration"""
    
    def __init__(self, input_dim=100, hidden_dim=64, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_sample_optimization_commits():
    """Create sample optimization commits with performance metrics"""
    
    commits = []
    
    # Commit 1: Model Architecture Optimization
    commit1 = OptimizationCommit(
        commit_id="opt_001",
        commit_hash="abc123def456",
        author="ML Engineer A",
        timestamp=datetime.now() - timedelta(days=5),
        message="Optimize model architecture with attention mechanism",
        commit_type=CommitType.OPTIMIZATION,
        status=CommitStatus.COMPLETED,
        model_size=2500000,  # 2.5M parameters
        inference_time=45.2,  # milliseconds
        memory_usage=2048,   # MB
        gpu_utilization=85.5,
        accuracy=0.923,
        loss=0.156,
        optimization_techniques=["attention_mechanism", "layer_norm", "residual_connections"],
        hyperparameters={"learning_rate": 0.001, "batch_size": 32, "epochs": 100},
        model_architecture="transformer",
        dataset_size=10000,
        training_epochs=50,
        experiment_id="exp_001",
        run_id="run_001",
        notes="Significant improvement in accuracy with attention mechanism"
    )
    commits.append(commit1)
    
    # Commit 2: Training Optimization
    commit2 = OptimizationCommit(
        commit_id="opt_002",
        commit_hash="def456ghi789",
        author="ML Engineer B",
        timestamp=datetime.now() - timedelta(days=3),
        message="Implement mixed precision training for faster convergence",
        commit_type=CommitType.TRAINING,
        status=CommitStatus.COMPLETED,
        model_size=2500000,
        inference_time=42.8,
        memory_usage=1536,
        gpu_utilization=92.3,
        accuracy=0.928,
        loss=0.142,
        optimization_techniques=["mixed_precision", "gradient_scaling", "autocast"],
        hyperparameters={"learning_rate": 0.001, "batch_size": 64, "epochs": 75},
        model_architecture="transformer",
        dataset_size=10000,
        training_epochs=75,
        experiment_id="exp_002",
        run_id="run_002",
        notes="Mixed precision training reduced memory usage by 25%"
    )
    commits.append(commit2)
    
    # Commit 3: Inference Optimization
    commit3 = OptimizationCommit(
        commit_id="opt_003",
        commit_hash="ghi789jkl012",
        author="ML Engineer C",
        timestamp=datetime.now() - timedelta(days=1),
        message="Implement model quantization for faster inference",
        commit_type=CommitType.INFERENCE,
        status=CommitStatus.COMPLETED,
        model_size=1250000,  # Quantized model is smaller
        inference_time=28.5,  # Much faster inference
        memory_usage=1024,
        gpu_utilization=78.2,
        accuracy=0.925,  # Slight accuracy drop due to quantization
        loss=0.148,
        optimization_techniques=["quantization", "int8_inference", "tensorrt"],
        hyperparameters={"quantization_bits": 8, "calibration_samples": 1000},
        model_architecture="transformer_quantized",
        dataset_size=10000,
        training_epochs=0,  # No training for quantization
        experiment_id="exp_003",
        run_id="run_003",
        notes="Quantization achieved 2x speedup with minimal accuracy loss"
    )
    commits.append(commit3)
    
    # Commit 4: Performance Optimization (Failed)
    commit4 = OptimizationCommit(
        commit_id="opt_004",
        commit_hash="jkl012mno345",
        author="ML Engineer A",
        timestamp=datetime.now() - timedelta(hours=6),
        message="Attempt model pruning for further optimization",
        commit_type=CommitType.PERFORMANCE,
        status=CommitStatus.FAILED,
        model_size=2500000,
        inference_time=0,  # Failed optimization
        memory_usage=0,
        gpu_utilization=0,
        accuracy=0.0,
        loss=0.0,
        optimization_techniques=["pruning", "magnitude_based"],
        hyperparameters={"pruning_ratio": 0.5, "sparsity": 0.8},
        model_architecture="transformer",
        dataset_size=10000,
        training_epochs=0,
        experiment_id="exp_004",
        run_id="run_004",
        notes="Pruning caused significant accuracy degradation"
    )
    commits.append(commit4)
    
    return commits

def demo_commit_tracking():
    """Demonstrate advanced commit tracking capabilities"""
    
    print("🚀 Advanced Commit Tracking System Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n📊 Initializing Components...")
    
    # Commit Tracker
    commit_tracker = create_commit_tracker(device="cpu", model_path="demo_commit_model.pth")
    
    # Version Manager
    version_manager = create_version_manager(
        base_path="demo_versions",
        use_wandb=False,  # Disable for demo
        use_tensorboard=False
    )
    
    # Optimization Registry
    optimization_registry = create_optimization_registry(
        registry_path="demo_registry",
        use_profiling=True,
        auto_benchmark=False  # Disable for demo
    )
    
    print("✅ Components initialized successfully")
    
    # Create sample commits
    print("\n📝 Creating Sample Optimization Commits...")
    sample_commits = create_sample_optimization_commits()
    
    # Add commits to tracker
    for commit in sample_commits:
        commit_tracker.add_commit(commit)
        print(f"  ✓ Added commit: {commit.commit_id} - {commit.message[:50]}...")
    
    print(f"✅ Added {len(sample_commits)} commits")
    
    # Demonstrate commit statistics
    print("\n📈 Commit Statistics:")
    stats = commit_tracker.get_performance_statistics()
    print(f"  Total commits: {stats['total_commits']}")
    print(f"  Commits with metrics: {stats['commits_with_metrics']}")
    print(f"  Average inference time: {stats['average_inference_time']:.2f}ms")
    print(f"  Average memory usage: {stats['average_memory_usage']:.2f}MB")
    print(f"  Average GPU utilization: {stats['average_gpu_utilization']:.2f}%")
    print(f"  Average accuracy: {stats['average_accuracy']:.3f}")
    print(f"  Best accuracy: {stats['best_accuracy']:.3f}")
    print(f"  Fastest inference: {stats['fastest_inference']:.2f}ms")
    
    # Demonstrate filtering
    print("\n🔍 Filtering Examples:")
    
    # Filter by author
    author_commits = commit_tracker.get_commits_by_author("ML Engineer A")
    print(f"  Commits by ML Engineer A: {len(author_commits)}")
    
    # Filter by commit type
    optimization_commits = commit_tracker.get_commits_by_type(CommitType.OPTIMIZATION)
    print(f"  Optimization commits: {len(optimization_commits)}")
    
    # Demonstrate performance prediction
    print("\n🔮 Performance Prediction:")
    
    # Create a sample commit for prediction
    sample_commit = OptimizationCommit(
        commit_id="pred_001",
        commit_hash="pred123",
        author="ML Engineer D",
        timestamp=datetime.now(),
        message="Sample commit for prediction",
        commit_type=CommitType.OPTIMIZATION,
        status=CommitStatus.PENDING,
        optimization_techniques=["new_technique"],
        hyperparameters={"learning_rate": 0.001}
    )
    
    # Train the performance predictor
    print("  Training performance predictor...")
    history = commit_tracker.train_performance_predictor(epochs=50, batch_size=8)
    
    if history:
        print(f"  Training completed. Final loss: {history['train_loss'][-1]:.4f}")
        
        # Make prediction
        predictions = commit_tracker.predict_performance(sample_commit)
        print(f"  Predicted inference time: {predictions['inference_time']:.2f}ms")
        print(f"  Predicted memory usage: {predictions['memory_usage']:.2f}MB")
        print(f"  Predicted accuracy: {predictions['accuracy']:.3f}")
    
    # Demonstrate optimization recommendations
    print("\n💡 Optimization Recommendations:")
    for commit in sample_commits[:3]:  # Show recommendations for first 3 commits
        recommendations = commit_tracker.get_optimization_recommendations(commit)
        print(f"  {commit.commit_id}: {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations[:2], 1):  # Show first 2 recommendations
            print(f"    {i}. {rec}")
    
    # Demonstrate version management
    print("\n📦 Version Management:")
    
    # Create versions
    version1 = create_version(
        version_manager,
        VersionType.MAJOR,
        "ML Engineer A",
        "Initial model with attention mechanism",
        model=SampleModel(),
        performance_metrics={"accuracy": 0.923, "inference_time": 45.2}
    )
    print(f"  Created version: {version1}")
    
    version2 = create_version(
        version_manager,
        VersionType.MINOR,
        "ML Engineer B",
        "Mixed precision training optimization",
        model=SampleModel(),
        performance_metrics={"accuracy": 0.928, "inference_time": 42.8}
    )
    print(f"  Created version: {version2}")
    
    # Get version history
    history = get_version_history(version_manager, limit=5)
    print(f"  Version history: {len(history)} versions")
    
    # Demonstrate optimization registry
    print("\n🔧 Optimization Registry:")
    
    # Register optimizations
    opt1 = register_optimization(
        optimization_registry,
        "Attention Mechanism",
        "Implements multi-head attention for better feature extraction",
        OptimizationCategory.MODEL_ARCHITECTURE,
        author="ML Engineer A",
        hyperparameters={"num_heads": 8, "d_model": 512}
    )
    print(f"  Registered optimization: {opt1}")
    
    opt2 = register_optimization(
        optimization_registry,
        "Mixed Precision Training",
        "Uses FP16 for faster training with minimal accuracy loss",
        OptimizationCategory.TRAINING_OPTIMIZATION,
        author="ML Engineer B",
        hyperparameters={"loss_scale": 1024, "autocast": True}
    )
    print(f"  Registered optimization: {opt2}")
    
    # Get registry statistics
    registry_stats = get_registry_statistics(optimization_registry)
    print(f"  Total optimizations: {registry_stats['total_entries']}")
    print(f"  Active optimizations: {registry_stats['active_entries']}")
    
    # Demonstrate benchmarking
    print("\n⚡ Performance Benchmarking:")
    
    # Create sample model and input
    model = SampleModel()
    input_data = torch.randn(32, 100)
    
    # Benchmark optimizations
    try:
        benchmark_results = optimization_registry.benchmark_optimization(
            opt1, model, input_data, iterations=10
        )
        print(f"  Benchmark completed for {opt1}")
        print(f"    Average time: {benchmark_results['average_time']:.4f}s")
        print(f"    Average memory: {benchmark_results['average_memory']:.2f}MB")
    except Exception as e:
        print(f"  Benchmark failed: {e}")
    
    # Demonstrate comparison
    print("\n📊 Optimization Comparison:")
    
    try:
        comparison = optimization_registry.compare_optimizations(
            [opt1, opt2], model, input_data
        )
        print(f"  Compared {len(comparison)} optimizations")
        for opt_id, results in comparison.items():
            print(f"    {opt_id}: {results['average_time']:.4f}s")
    except Exception as e:
        print(f"  Comparison failed: {e}")
    
    # Final summary
    print("\n🎯 System Summary:")
    print(f"  Commit tracker: {len(commit_tracker.commits)} commits")
    print(f"  Version manager: {len(version_manager.versions)} versions")
    print(f"  Optimization registry: {len(optimization_registry.entries)} optimizations")
    
    print("\n✅ Demo completed successfully!")
    
    # Cleanup
    print("\n🧹 Cleaning up demo files...")
    import shutil
    
    try:
        if os.path.exists("demo_commit_model.pth"):
            os.remove("demo_commit_model.pth")
        if os.path.exists("demo_versions"):
            shutil.rmtree("demo_versions")
        if os.path.exists("demo_registry"):
            shutil.rmtree("demo_registry")
        print("  Cleanup completed")
    except Exception as e:
        print(f"  Cleanup warning: {e}")

def demo_advanced_features():
    """Demonstrate advanced features"""
    
    print("\n🚀 Advanced Features Demo")
    print("=" * 30)
    
    # Initialize tracker
    tracker = create_commit_tracker()
    
    # Create commits with different performance characteristics
    commits = [
        OptimizationCommit(
            commit_id="perf_001",
            commit_hash="perf001",
            author="Engineer A",
            timestamp=datetime.now(),
            message="High performance optimization",
            commit_type=CommitType.OPTIMIZATION,
            status=CommitStatus.COMPLETED,
            inference_time=25.0,
            memory_usage=1024,
            gpu_utilization=95.0,
            accuracy=0.95,
            loss=0.1,
            optimization_techniques=["quantization", "pruning"]
        ),
        OptimizationCommit(
            commit_id="perf_002",
            commit_hash="perf002",
            author="Engineer B",
            timestamp=datetime.now(),
            message="Memory optimization",
            commit_type=CommitType.OPTIMIZATION,
            status=CommitStatus.COMPLETED,
            inference_time=35.0,
            memory_usage=512,
            gpu_utilization=80.0,
            accuracy=0.92,
            loss=0.15,
            optimization_techniques=["gradient_checkpointing", "mixed_precision"]
        )
    ]
    
    # Add commits
    for commit in commits:
        tracker.add_commit(commit)
    
    # Demonstrate performance analysis
    print("📊 Performance Analysis:")
    stats = tracker.get_performance_statistics()
    
    print(f"  Best accuracy: {stats['best_accuracy']:.3f}")
    print(f"  Fastest inference: {stats['fastest_inference']:.2f}ms")
    print(f"  Lowest memory: {stats['lowest_memory']:.2f}MB")
    
    # Demonstrate recommendations
    print("\n💡 Smart Recommendations:")
    for commit in commits:
        recommendations = tracker.get_optimization_recommendations(commit)
        print(f"  {commit.commit_id}: {len(recommendations)} recommendations")
        for rec in recommendations:
            print(f"    - {rec}")
    
    print("\n✅ Advanced features demo completed!")

if __name__ == "__main__":
    try:
        # Run main demo
        demo_commit_tracking()
        
        # Run advanced features demo
        demo_advanced_features()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



