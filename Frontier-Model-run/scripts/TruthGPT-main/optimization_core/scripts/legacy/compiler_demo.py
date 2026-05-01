"""
TruthGPT Compiler Integration Demo
Demonstrates the integration between compiler infrastructure and TruthGPT optimizers
"""

import logging
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import TruthGPT components
from .compiler_integration import (
    TruthGPTCompilerIntegration, TruthGPTCompilationConfig, TruthGPTCompilationResult,
    create_truthgpt_compiler_integration, truthgpt_compilation_context
)

from .compiler import (
    CompilationTarget, OptimizationLevel, CompilationConfig,
    create_compiler_core, compilation_context
)

# Import unified TruthGPT optimizers
try:
    from .optimizers import create_truthgpt_optimizer
    TRUTHGPT_OPTIMIZERS_AVAILABLE = True
except ImportError:
    logger.warning("TruthGPT optimizers not available, using mock optimizers")
    TRUTHGPT_OPTIMIZERS_AVAILABLE = False

class MockTruthGPTOptimizer:
    """Mock TruthGPT optimizer for demonstration purposes"""
    
    def __init__(self, name: str = "MockOptimizer"):
        self.name = name
        self.optimization_count = 0
        
    def optimize(self, model: Any) -> Any:
        """Mock optimization method"""
        self.optimization_count += 1
        logger.info(f"{self.name} optimized model (optimization #{self.optimization_count})")
        return model
    
    def enhance_model(self, model: Any) -> Any:
        """Mock model enhancement method"""
        logger.info(f"{self.name} enhanced model")
        return model

class SimpleTransformerModel(nn.Module):
    """Simple transformer model for demonstration"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2048, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:seq_len]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_projection(x)

def create_demo_model() -> SimpleTransformerModel:
    """Create a demo model for testing"""
    logger.info("Creating demo transformer model")
    model = SimpleTransformerModel(vocab_size=10000, d_model=512, n_heads=8, n_layers=6)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    return model

def create_demo_optimizers() -> Dict[str, Any]:
    """Create demo optimizers"""
    optimizers = {}
    
    if TRUTHGPT_OPTIMIZERS_AVAILABLE:
        try:
            optimizers["ultimate"] = create_truthgpt_optimizer("ultimate")
            logger.info("Created Ultimate TruthGPT Optimizer")
        except Exception as e:
            logger.warning(f"Failed to create Ultimate TruthGPT Optimizer: {e}")
        
        try:
            optimizers["transcendent"] = create_truthgpt_optimizer("transcendent")
            logger.info("Created Transcendent TruthGPT Optimizer")
        except Exception as e:
            logger.warning(f"Failed to create Transcendent TruthGPT Optimizer: {e}")
        
        try:
            optimizers["infinite"] = create_truthgpt_optimizer("infinite")
            logger.info("Created Infinite TruthGPT Optimizer")
        except Exception as e:
            logger.warning(f"Failed to create Infinite TruthGPT Optimizer: {e}")
    
    # Add mock optimizers if needed
    if not optimizers:
        optimizers["mock_ultimate"] = MockTruthGPTOptimizer("Mock Ultimate")
        optimizers["mock_transcendent"] = MockTruthGPTOptimizer("Mock Transcendent")
        optimizers["mock_infinite"] = MockTruthGPTOptimizer("Mock Infinite")
        logger.info("Created mock optimizers for demonstration")
    
    return optimizers

def demo_basic_compilation():
    """Demonstrate basic compilation functionality"""
    logger.info("=" * 60)
    logger.info("DEMO: Basic Compilation")
    logger.info("=" * 60)
    
    # Create demo model
    model = create_demo_model()
    
    # Create compilation configuration
    config = CompilationConfig(
        target=CompilationTarget.CPU,
        optimization_level=OptimizationLevel.STANDARD,
        enable_quantization=True,
        enable_fusion=True
    )
    
    # Create compiler
    compiler = create_compiler_core(config)
    
    # Compile model
    with compilation_context(config) as ctx:
        result = compiler.compile(model)
    
    if result.success:
        logger.info(f"✅ Compilation successful in {result.compilation_time:.3f}s")
        logger.info(f"Optimization metrics: {result.optimization_metrics}")
    else:
        logger.error(f"❌ Compilation failed: {result.errors}")
    
    return result

def demo_truthgpt_compiler_integration():
    """Demonstrate TruthGPT compiler integration"""
    logger.info("=" * 60)
    logger.info("DEMO: TruthGPT Compiler Integration")
    logger.info("=" * 60)
    
    # Create demo model and optimizers
    model = create_demo_model()
    optimizers = create_demo_optimizers()
    
    # Create TruthGPT compilation configuration
    config = TruthGPTCompilationConfig(
        primary_compiler="aot",
        fallback_compilers=["jit", "mlir", "runtime"],
        optimization_level=OptimizationLevel.EXTREME,
        target_platform=CompilationTarget.GPU,
        enable_truthgpt_optimizations=True,
        enable_profiling=True,
        enable_benchmarking=True,
        auto_select_compiler=True
    )
    
    # Create compiler integration
    integration = create_truthgpt_compiler_integration(config)
    
    # Test with different optimizers
    for optimizer_name, optimizer in optimizers.items():
        logger.info(f"\n--- Testing with {optimizer_name} optimizer ---")
        
        try:
            result = integration.compile_truthgpt_model(model, optimizer)
            
            if result.success:
                logger.info(f"✅ Compilation successful with {result.primary_compiler_used} compiler")
                logger.info(f"Compilation time: {result.integration_metadata.get('compilation_time', 0):.3f}s")
                logger.info(f"Performance metrics: {result.performance_metrics}")
                
                # Show optimization report
                if result.optimization_report:
                    report = result.optimization_report
                    logger.info(f"Model size: {report['model_info']['estimated_size']:,} parameters")
                    logger.info(f"Best compiler: {report['compilation_summary']['best_compiler']}")
            else:
                logger.error(f"❌ Compilation failed with {optimizer_name}")
                
        except Exception as e:
            logger.error(f"❌ Error with {optimizer_name}: {e}")
    
    return integration

def demo_compiler_benchmarking():
    """Demonstrate compiler benchmarking"""
    logger.info("=" * 60)
    logger.info("DEMO: Compiler Benchmarking")
    logger.info("=" * 60)
    
    # Create demo model
    model = create_demo_model()
    
    # Create TruthGPT compilation configuration
    config = TruthGPTCompilationConfig(
        optimization_level=OptimizationLevel.STANDARD,
        target_platform=CompilationTarget.CPU,
        enable_benchmarking=True
    )
    
    # Create compiler integration
    integration = create_truthgpt_compiler_integration(config)
    
    # Run benchmarks
    logger.info("Running compiler benchmarks...")
    benchmark_results = integration.benchmark_compilers(model, iterations=3)
    
    # Display results
    logger.info("\n📊 Benchmark Results:")
    for compiler_name, results in benchmark_results.items():
        logger.info(f"\n{compiler_name.upper()} Compiler:")
        logger.info(f"  Average time: {results['avg_time']:.3f}s")
        logger.info(f"  Min time: {results['min_time']:.3f}s")
        logger.info(f"  Max time: {results['max_time']:.3f}s")
        logger.info(f"  Success rate: {results['success_rate']:.1%}")
    
    # Get compiler statistics
    stats = integration.get_compiler_statistics()
    logger.info(f"\n📈 Compiler Statistics:")
    logger.info(f"  Available compilers: {stats['available_compilers']}")
    logger.info(f"  Total compilations: {stats['total_compilations']}")
    
    return benchmark_results

def demo_advanced_compilation_scenarios():
    """Demonstrate advanced compilation scenarios"""
    logger.info("=" * 60)
    logger.info("DEMO: Advanced Compilation Scenarios")
    logger.info("=" * 60)
    
    # Create demo model
    model = create_demo_model()
    
    # Scenario 1: GPU optimization with TensorRT
    logger.info("\n--- Scenario 1: GPU Optimization with TensorRT ---")
    try:
        config = TruthGPTCompilationConfig(
            primary_compiler="tensorrt",
            target_platform=CompilationTarget.GPU,
            optimization_level=OptimizationLevel.EXTREME,
            enable_truthgpt_optimizations=True
        )
        
        integration = create_truthgpt_compiler_integration(config)
        result = integration.compile_truthgpt_model(model)
        
        if result.success:
            logger.info(f"✅ TensorRT compilation successful")
            logger.info(f"Performance metrics: {result.performance_metrics}")
        else:
            logger.warning(f"⚠️ TensorRT compilation failed, trying fallback")
            
    except Exception as e:
        logger.warning(f"⚠️ TensorRT not available: {e}")
    
    # Scenario 2: MLIR compilation
    logger.info("\n--- Scenario 2: MLIR Compilation ---")
    try:
        config = TruthGPTCompilationConfig(
            primary_compiler="mlir",
            target_platform=CompilationTarget.CPU,
            optimization_level=OptimizationLevel.AGGRESSIVE
        )
        
        integration = create_truthgpt_compiler_integration(config)
        result = integration.compile_truthgpt_model(model)
        
        if result.success:
            logger.info(f"✅ MLIR compilation successful")
            if hasattr(result.compilation_results.get('mlir', {}), 'mlir_ir'):
                logger.info("MLIR IR generated successfully")
        else:
            logger.warning(f"⚠️ MLIR compilation failed")
            
    except Exception as e:
        logger.warning(f"⚠️ MLIR compilation error: {e}")
    
    # Scenario 3: Adaptive compilation with JIT
    logger.info("\n--- Scenario 3: Adaptive JIT Compilation ---")
    try:
        config = TruthGPTCompilationConfig(
            primary_compiler="jit",
            target_platform=CompilationTarget.CPU,
            optimization_level=OptimizationLevel.ADAPTIVE,
            enable_adaptive_compilation=True
        )
        
        integration = create_truthgpt_compiler_integration(config)
        
        # Simulate multiple executions for JIT optimization
        for i in range(5):
            result = integration.compile_truthgpt_model(model)
            if result.success:
                logger.info(f"JIT execution {i+1}: {result.performance_metrics.get('total_compilation_time', 0):.3f}s")
        
        logger.info("✅ JIT adaptive compilation completed")
        
    except Exception as e:
        logger.warning(f"⚠️ JIT compilation error: {e}")

def demo_compiler_context_usage():
    """Demonstrate compiler context usage"""
    logger.info("=" * 60)
    logger.info("DEMO: Compiler Context Usage")
    logger.info("=" * 60)
    
    # Create demo model
    model = create_demo_model()
    
    # Create configuration
    config = TruthGPTCompilationConfig(
        primary_compiler="aot",
        optimization_level=OptimizationLevel.STANDARD,
        enable_profiling=True
    )
    
    # Use context manager
    with truthgpt_compilation_context(config) as integration:
        logger.info("Compiler integration context started")
        
        # Compile model
        result = integration.compile_truthgpt_model(model)
        
        if result.success:
            logger.info(f"✅ Context-based compilation successful")
            logger.info(f"Used compiler: {result.primary_compiler_used}")
        else:
            logger.error(f"❌ Context-based compilation failed")
    
    logger.info("Compiler integration context ended")

def main():
    """Main demo function"""
    logger.info("🚀 Starting TruthGPT Compiler Integration Demo")
    logger.info("=" * 80)
    
    try:
        # Run all demos
        demo_basic_compilation()
        demo_truthgpt_compiler_integration()
        demo_compiler_benchmarking()
        demo_advanced_compilation_scenarios()
        demo_compiler_context_usage()
        
        logger.info("=" * 80)
        logger.info("🎉 TruthGPT Compiler Integration Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()






