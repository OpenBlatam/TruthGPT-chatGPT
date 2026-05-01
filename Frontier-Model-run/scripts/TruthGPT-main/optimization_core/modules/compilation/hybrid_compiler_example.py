"""
TruthGPT Hybrid Compiler Integration Example
Demonstrates advanced hybrid compilation combining Neural, Quantum, and Transcendent optimizations
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import hybrid compiler integration
from .hybrid_compiler_integration import (
    HybridCompilerIntegration, HybridCompilationConfig, HybridCompilationResult,
    HybridCompilationStrategy, HybridOptimizationMode,
    create_hybrid_compiler_integration, hybrid_compilation_context
)

def create_example_model() -> nn.Module:
    """Create an example model for compilation."""
    return nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(250, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

def create_large_model() -> nn.Module:
    """Create a large model for advanced compilation."""
    return nn.Sequential(
        nn.Linear(10000, 5000),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(5000, 2500),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2500, 1000),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

def create_transformer_model() -> nn.Module:
    """Create a transformer-like model."""
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=512, nhead=8, num_layers=6):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(10000, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=2048
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.classifier = nn.Linear(d_model, 10)
        
        def forward(self, x):
            x = self.embedding(x) + self.pos_encoding[:x.size(1)]
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)
    
    return SimpleTransformer()

def example_fusion_compilation():
    """Example of fusion compilation combining all compilers."""
    logger.info("=== Fusion Compilation Example ===")
    
    try:
        # Create configuration for fusion compilation
        config = HybridCompilationConfig(
            target="cuda" if torch.cuda.is_available() else "cpu",
            compilation_strategy=HybridCompilationStrategy.FUSION,
            optimization_mode=HybridOptimizationMode.BALANCED,
            enable_neural_compilation=True,
            enable_quantum_compilation=True,
            enable_transcendent_compilation=True,
            fusion_weight_neural=0.4,
            fusion_weight_quantum=0.3,
            fusion_weight_transcendent=0.3,
            enable_profiling=True,
            enable_monitoring=True
        )
        
        # Create integration
        integration = create_hybrid_compiler_integration(config)
        
        # Create example model
        model = create_example_model()
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Compile model
        start_time = time.time()
        result = integration.compile(model)
        compilation_time = time.time() - start_time
        
        # Display results
        if result.success:
            logger.info(f"Fusion compilation successful!")
            logger.info(f"Compilation time: {compilation_time:.3f}s")
            logger.info(f"Hybrid efficiency: {result.hybrid_efficiency:.3f}")
            logger.info(f"Neural contribution: {result.neural_contribution:.3f}")
            logger.info(f"Quantum contribution: {result.quantum_contribution:.3f}")
            logger.info(f"Transcendent contribution: {result.transcendent_contribution:.3f}")
            logger.info(f"Fusion score: {result.fusion_score:.3f}")
            logger.info(f"Optimizations applied: {result.optimization_applied}")
            logger.info(f"Component results: {list(result.component_results.keys())}")
            
            # Display performance metrics
            if result.performance_metrics:
                logger.info("Performance metrics:")
                for key, value in result.performance_metrics.items():
                    logger.info(f"  {key}: {value}")
            
            # Display hybrid states
            if result.hybrid_states:
                logger.info("Hybrid states:")
                for key, value in result.hybrid_states.items():
                    logger.info(f"  {key}: {value}")
        else:
            logger.error(f"Fusion compilation failed: {result.errors}")
        
        return result
        
    except Exception as e:
        logger.error(f"Fusion compilation example failed: {e}")
        raise

def example_adaptive_compilation():
    """Example of adaptive compilation with intelligent compiler selection."""
    logger.info("=== Adaptive Compilation Example ===")
    
    try:
        # Create configuration for adaptive compilation
        config = HybridCompilationConfig(
            target="cuda" if torch.cuda.is_available() else "cpu",
            compilation_strategy=HybridCompilationStrategy.ADAPTIVE,
            optimization_mode=HybridOptimizationMode.INTELLIGENT,
            enable_neural_compilation=True,
            enable_quantum_compilation=True,
            enable_transcendent_compilation=True,
            enable_adaptive_selection=True,
            model_analysis_depth=5,
            performance_prediction=True,
            enable_profiling=True
        )
        
        # Create integration
        integration = create_hybrid_compiler_integration(config)
        
        # Test with different model sizes
        models = [
            ("Small Model", create_example_model()),
            ("Large Model", create_large_model()),
            ("Transformer Model", create_transformer_model())
        ]
        
        for model_name, model in models:
            logger.info(f"\nTesting {model_name}:")
            logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
            
            # Compile model
            start_time = time.time()
            result = integration.compile(model)
            compilation_time = time.time() - start_time
            
            if result.success:
                logger.info(f"Adaptive compilation successful!")
                logger.info(f"Compilation time: {compilation_time:.3f}s")
                logger.info(f"Hybrid efficiency: {result.hybrid_efficiency:.3f}")
                logger.info(f"Selected compiler: {result.component_results.get('selected_compiler', 'unknown')}")
                logger.info(f"Strategy: {result.component_results.get('strategy', 'unknown')}")
            else:
                logger.error(f"Adaptive compilation failed: {result.errors}")
        
        return True
        
    except Exception as e:
        logger.error(f"Adaptive compilation example failed: {e}")
        raise

def example_parallel_compilation():
    """Example of parallel compilation with multiple compilers."""
    logger.info("=== Parallel Compilation Example ===")
    
    try:
        # Create configuration for parallel compilation
        config = HybridCompilationConfig(
            target="cuda" if torch.cuda.is_available() else "cpu",
            compilation_strategy=HybridCompilationStrategy.PARALLEL,
            optimization_mode=HybridOptimizationMode.BALANCED,
            enable_neural_compilation=True,
            enable_quantum_compilation=True,
            enable_transcendent_compilation=True,
            enable_parallel_compilation=True,
            max_parallel_workers=4,
            enable_profiling=True
        )
        
        # Create integration
        integration = create_hybrid_compiler_integration(config)
        
        # Create example model
        model = create_large_model()
        logger.info(f"Created large model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Compile model
        start_time = time.time()
        result = integration.compile(model)
        compilation_time = time.time() - start_time
        
        # Display results
        if result.success:
            logger.info(f"Parallel compilation successful!")
            logger.info(f"Compilation time: {compilation_time:.3f}s")
            logger.info(f"Hybrid efficiency: {result.hybrid_efficiency:.3f}")
            logger.info(f"Fusion score: {result.fusion_score:.3f}")
            
            # Display component results
            logger.info("Component results:")
            for compiler_name, compiler_result in result.component_results.items():
                if hasattr(compiler_result, 'success'):
                    logger.info(f"  {compiler_name}: {'Success' if compiler_result.success else 'Failed'}")
                    if hasattr(compiler_result, 'compilation_time'):
                        logger.info(f"    Compilation time: {compiler_result.compilation_time:.3f}s")
        else:
            logger.error(f"Parallel compilation failed: {result.errors}")
        
        return result
        
    except Exception as e:
        logger.error(f"Parallel compilation example failed: {e}")
        raise

def example_cascade_compilation():
    """Example of cascade compilation with quality thresholds."""
    logger.info("=== Cascade Compilation Example ===")
    
    try:
        # Create configuration for cascade compilation
        config = HybridCompilationConfig(
            target="cuda" if torch.cuda.is_available() else "cpu",
            compilation_strategy=HybridCompilationStrategy.CASCADE,
            optimization_mode=HybridOptimizationMode.BALANCED,
            enable_neural_compilation=True,
            enable_quantum_compilation=True,
            enable_transcendent_compilation=True,
            cascade_order=["neural", "quantum", "transcendent"],
            cascade_threshold=0.7,
            enable_profiling=True
        )
        
        # Create integration
        integration = create_hybrid_compiler_integration(config)
        
        # Create example model
        model = create_transformer_model()
        logger.info(f"Created transformer model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Compile model
        start_time = time.time()
        result = integration.compile(model)
        compilation_time = time.time() - start_time
        
        # Display results
        if result.success:
            logger.info(f"Cascade compilation successful!")
            logger.info(f"Compilation time: {compilation_time:.3f}s")
            logger.info(f"Hybrid efficiency: {result.hybrid_efficiency:.3f}")
            logger.info(f"Fusion score: {result.fusion_score:.3f}")
            
            # Display cascade results
            logger.info("Cascade results:")
            for compiler_name, compiler_result in result.component_results.items():
                if hasattr(compiler_result, 'success'):
                    logger.info(f"  {compiler_name}: {'Success' if compiler_result.success else 'Failed'}")
                    if hasattr(compiler_result, 'neural_accuracy'):
                        logger.info(f"    Neural accuracy: {compiler_result.neural_accuracy:.3f}")
                    if hasattr(compiler_result, 'quantum_fidelity'):
                        logger.info(f"    Quantum fidelity: {compiler_result.quantum_fidelity:.3f}")
                    if hasattr(compiler_result, 'consciousness_level'):
                        logger.info(f"    Consciousness level: {compiler_result.consciousness_level:.3f}")
        else:
            logger.error(f"Cascade compilation failed: {result.errors}")
        
        return result
        
    except Exception as e:
        logger.error(f"Cascade compilation example failed: {e}")
        raise

def example_hierarchical_compilation():
    """Example of hierarchical compilation with multiple levels."""
    logger.info("=== Hierarchical Compilation Example ===")
    
    try:
        # Create configuration for hierarchical compilation
        config = HybridCompilationConfig(
            target="cuda" if torch.cuda.is_available() else "cpu",
            compilation_strategy=HybridCompilationStrategy.HIERARCHICAL,
            optimization_mode=HybridOptimizationMode.BALANCED,
            enable_neural_compilation=True,
            enable_quantum_compilation=True,
            enable_transcendent_compilation=True,
            hierarchy_levels=3,
            level_weights=[0.5, 0.3, 0.2],
            enable_profiling=True
        )
        
        # Create integration
        integration = create_hybrid_compiler_integration(config)
        
        # Create example model
        model = create_large_model()
        logger.info(f"Created large model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Compile model
        start_time = time.time()
        result = integration.compile(model)
        compilation_time = time.time() - start_time
        
        # Display results
        if result.success:
            logger.info(f"Hierarchical compilation successful!")
            logger.info(f"Compilation time: {compilation_time:.3f}s")
            logger.info(f"Hybrid efficiency: {result.hybrid_efficiency:.3f}")
            logger.info(f"Fusion score: {result.fusion_score:.3f}")
            
            # Display hierarchical results
            logger.info("Hierarchical results:")
            for compiler_name, compiler_result in result.component_results.items():
                if hasattr(compiler_result, 'success'):
                    logger.info(f"  {compiler_name}: {'Success' if compiler_result.success else 'Failed'}")
                    if hasattr(compiler_result, 'compilation_time'):
                        logger.info(f"    Compilation time: {compiler_result.compilation_time:.3f}s")
        else:
            logger.error(f"Hierarchical compilation failed: {result.errors}")
        
        return result
        
    except Exception as e:
        logger.error(f"Hierarchical compilation example failed: {e}")
        raise

def example_performance_comparison():
    """Example comparing different compilation strategies."""
    logger.info("=== Performance Comparison Example ===")
    
    try:
        # Create example model
        model = create_large_model()
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test different strategies
        strategies = [
            HybridCompilationStrategy.FUSION,
            HybridCompilationStrategy.ADAPTIVE,
            HybridCompilationStrategy.PARALLEL,
            HybridCompilationStrategy.CASCADE,
            HybridCompilationStrategy.HIERARCHICAL
        ]
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"\nTesting {strategy.value} strategy:")
            
            # Create configuration
            config = HybridCompilationConfig(
                target="cuda" if torch.cuda.is_available() else "cpu",
                compilation_strategy=strategy,
                optimization_mode=HybridOptimizationMode.BALANCED,
                enable_neural_compilation=True,
                enable_quantum_compilation=True,
                enable_transcendent_compilation=True,
                enable_profiling=True
            )
            
            # Create integration
            integration = create_hybrid_compiler_integration(config)
            
            # Compile model
            start_time = time.time()
            result = integration.compile(model)
            compilation_time = time.time() - start_time
            
            if result.success:
                results[strategy.value] = {
                    'compilation_time': compilation_time,
                    'hybrid_efficiency': result.hybrid_efficiency,
                    'fusion_score': result.fusion_score,
                    'neural_contribution': result.neural_contribution,
                    'quantum_contribution': result.quantum_contribution,
                    'transcendent_contribution': result.transcendent_contribution
                }
                
                logger.info(f"  Compilation time: {compilation_time:.3f}s")
                logger.info(f"  Hybrid efficiency: {result.hybrid_efficiency:.3f}")
                logger.info(f"  Fusion score: {result.fusion_score:.3f}")
            else:
                logger.error(f"  Compilation failed: {result.errors}")
        
        # Display comparison
        logger.info("\n=== Performance Comparison Summary ===")
        logger.info("Strategy\t\tTime(s)\tEfficiency\tFusion\tNeural\tQuantum\tTranscendent")
        logger.info("-" * 80)
        
        for strategy_name, metrics in results.items():
            logger.info(f"{strategy_name:<20}\t{metrics['compilation_time']:.3f}\t"
                       f"{metrics['hybrid_efficiency']:.3f}\t\t{metrics['fusion_score']:.3f}\t"
                       f"{metrics['neural_contribution']:.3f}\t{metrics['quantum_contribution']:.3f}\t"
                       f"{metrics['transcendent_contribution']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Performance comparison example failed: {e}")
        raise

def example_context_usage():
    """Example of using hybrid compilation context."""
    logger.info("=== Context Usage Example ===")
    
    try:
        # Create configuration
        config = HybridCompilationConfig(
            target="cuda" if torch.cuda.is_available() else "cpu",
            compilation_strategy=HybridCompilationStrategy.FUSION,
            optimization_mode=HybridOptimizationMode.BALANCED,
            enable_neural_compilation=True,
            enable_quantum_compilation=True,
            enable_transcendent_compilation=True
        )
        
        # Use context manager
        with hybrid_compilation_context(config) as integration:
            # Create example model
            model = create_example_model()
            logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
            
            # Compile model
            result = integration.compile(model)
            
            if result.success:
                logger.info(f"Context compilation successful!")
                logger.info(f"Hybrid efficiency: {result.hybrid_efficiency:.3f}")
                logger.info(f"Fusion score: {result.fusion_score:.3f}")
            else:
                logger.error(f"Context compilation failed: {result.errors}")
        
        return True
        
    except Exception as e:
        logger.error(f"Context usage example failed: {e}")
        raise

def run_all_examples():
    """Run all hybrid compilation examples."""
    logger.info("Starting TruthGPT Hybrid Compiler Integration Examples")
    logger.info("=" * 60)
    
    try:
        # Run examples
        example_fusion_compilation()
        example_adaptive_compilation()
        example_parallel_compilation()
        example_cascade_compilation()
        example_hierarchical_compilation()
        example_performance_comparison()
        example_context_usage()
        
        logger.info("\n" + "=" * 60)
        logger.info("All hybrid compilation examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Hybrid compilation examples failed: {e}")
        raise

if __name__ == "__main__":
    # Run all examples
    run_all_examples()


