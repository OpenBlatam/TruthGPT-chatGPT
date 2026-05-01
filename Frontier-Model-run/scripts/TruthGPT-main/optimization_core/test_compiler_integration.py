"""
TruthGPT Compiler Integration Tests
Comprehensive test suite for compiler infrastructure integration
"""

import unittest
import logging
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
logger = logging.getLogger(__name__)

# Import components to test
try:
    from modules.compilation.compiler_integration import (
        TruthGPTCompilerIntegration, TruthGPTCompilationConfig, TruthGPTCompilationResult,
        create_truthgpt_compiler_integration
    )

    from modules.compilation.core_compiler.compiler import (
        CompilationTarget, OptimizationLevel, CompilationConfig,
        create_compiler_core, CompilationResult
    )
except ImportError:
    # Fallback for different import paths
    from .modules.compilation.compiler_integration import (
        TruthGPTCompilerIntegration, TruthGPTCompilationConfig, TruthGPTCompilationResult,
        create_truthgpt_compiler_integration
    )

    from .modules.compilation.core_compiler.compiler import (
        CompilationTarget, OptimizationLevel, CompilationConfig,
        create_compiler_core, CompilationResult
    )

class TestModel(nn.Module):
    """Simple test model for compiler testing"""
    
    def __init__(self, input_size: int = 100, hidden_size: int = 50, output_size: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class MockOptimizer:
    """Mock optimizer for testing"""
    
    def __init__(self, name: str = "MockOptimizer"):
        self.name = name
        self.optimization_count = 0
        
    def optimize(self, model: Any) -> Any:
        self.optimization_count += 1
        return model

class TestTruthGPTCompilerIntegration(unittest.TestCase):
    """Test suite for TruthGPT compiler integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_model = TestModel()
        self.mock_optimizer = MockOptimizer("TestOptimizer")
        
        # Create test configuration
        self.test_config = TruthGPTCompilationConfig(
            primary_compiler="aot",
            fallback_compilers=["jit", "mlir"],
            optimization_level=OptimizationLevel.STANDARD,
            target_platform=CompilationTarget.CPU,
            enable_truthgpt_optimizations=True,
            enable_profiling=True,
            enable_benchmarking=True,
            auto_select_compiler=True
        )
    
    def test_compiler_integration_creation(self):
        """Test compiler integration creation"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        self.assertIsInstance(integration, TruthGPTCompilerIntegration)
        self.assertIsNotNone(integration.compilers)
        self.assertIsInstance(integration.compilers, dict)
        
        # Check that at least some compilers were initialized
        self.assertGreater(len(integration.compilers), 0)
        
        logger.info(f"Created integration with {len(integration.compilers)} compilers: {list(integration.compilers.keys())}")
    
    def test_compilation_without_optimizer(self):
        """Test compilation without TruthGPT optimizer"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        result = integration.compile_truthgpt_model(self.test_model)
        
        self.assertIsInstance(result, TruthGPTCompilationResult)
        self.assertIsNotNone(result.primary_compiler_used)
        self.assertIsInstance(result.compilation_results, dict)
        self.assertIsInstance(result.performance_metrics, dict)
        self.assertIsInstance(result.optimization_report, dict)
        
        logger.info(f"Compilation without optimizer: {result.primary_compiler_used}, success: {result.success}")
    
    def test_compilation_with_optimizer(self):
        """Test compilation with TruthGPT optimizer"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        result = integration.compile_truthgpt_model(self.test_model, self.mock_optimizer)
        
        self.assertIsInstance(result, TruthGPTCompilationResult)
        self.assertIsNotNone(result.primary_compiler_used)
        self.assertIsInstance(result.compilation_results, dict)
        
        # Check that optimizer was called
        self.assertGreater(self.mock_optimizer.optimization_count, 0)
        
        logger.info(f"Compilation with optimizer: {result.primary_compiler_used}, success: {result.success}")
    
    def test_compiler_selection(self):
        """Test automatic compiler selection"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        # Test with different model sizes
        small_model = TestModel(10, 5, 2)
        large_model = TestModel(1000, 500, 100)
        
        # Compile both models
        small_result = integration.compile_truthgpt_model(small_model)
        large_result = integration.compile_truthgpt_model(large_model)
        
        self.assertIsNotNone(small_result.primary_compiler_used)
        self.assertIsNotNone(large_result.primary_compiler_used)
        
        logger.info(f"Small model compiler: {small_result.primary_compiler_used}")
        logger.info(f"Large model compiler: {large_result.primary_compiler_used}")
    
    def test_fallback_compilation(self):
        """Test fallback compilation when primary compiler fails"""
        # Create config with invalid primary compiler
        config = TruthGPTCompilationConfig(
            primary_compiler="nonexistent",
            fallback_compilers=["aot", "jit"],
            optimization_level=OptimizationLevel.STANDARD,
            target_platform=CompilationTarget.CPU
        )
        
        integration = create_truthgpt_compiler_integration(config)
        
        result = integration.compile_truthgpt_model(self.test_model)
        
        # Should still work with fallback compilers
        self.assertIsInstance(result, TruthGPTCompilationResult)
        self.assertIsNotNone(result.primary_compiler_used)
        
        logger.info(f"Fallback compilation: {result.primary_compiler_used}, success: {result.success}")
    
    def test_performance_metrics(self):
        """Test performance metrics generation"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        result = integration.compile_truthgpt_model(self.test_model)
        
        # Check performance metrics
        self.assertIsInstance(result.performance_metrics, dict)
        self.assertIn("total_compilation_time", result.performance_metrics)
        self.assertIn("compilers_used", result.performance_metrics)
        
        # Check optimization report
        self.assertIsInstance(result.optimization_report, dict)
        self.assertIn("model_info", result.optimization_report)
        self.assertIn("compilation_summary", result.optimization_report)
        
        logger.info(f"Performance metrics: {result.performance_metrics}")
    
    def test_compiler_statistics(self):
        """Test compiler statistics collection"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        # Run some compilations
        for i in range(3):
            integration.compile_truthgpt_model(self.test_model)
        
        stats = integration.get_compiler_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("available_compilers", stats)
        self.assertIn("total_compilations", stats)
        self.assertIn("performance_history", stats)
        
        self.assertGreaterEqual(stats["total_compilations"], 3)
        
        logger.info(f"Compiler statistics: {stats}")
    
    def test_benchmarking(self):
        """Test compiler benchmarking"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        # Run benchmarks with fewer iterations for speed
        benchmark_results = integration.benchmark_compilers(self.test_model, iterations=2)
        
        self.assertIsInstance(benchmark_results, dict)
        
        for compiler_name, results in benchmark_results.items():
            self.assertIn("avg_time", results)
            self.assertIn("success_rate", results)
            self.assertIn("total_iterations", results)
            
            self.assertEqual(results["total_iterations"], 2)
        
        logger.info(f"Benchmark results: {list(benchmark_results.keys())}")
    
    def test_compilation_context(self):
        """Test compilation context usage"""
        from .compiler_integration import truthgpt_compilation_context
        
        with truthgpt_compilation_context(self.test_config) as integration:
            self.assertIsInstance(integration, TruthGPTCompilerIntegration)
            
            result = integration.compile_truthgpt_model(self.test_model)
            self.assertIsInstance(result, TruthGPTCompilationResult)
    
    def test_error_handling(self):
        """Test error handling in compilation"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        # Test with None model
        result = integration.compile_truthgpt_model(None)
        
        self.assertIsInstance(result, TruthGPTCompilationResult)
        self.assertFalse(result.success)
        self.assertIn("error", result.integration_metadata)
        
        logger.info(f"Error handling test: {result.integration_metadata.get('error', 'No error')}")
    
    def test_optimization_cache(self):
        """Test optimization caching"""
        integration = create_truthgpt_compiler_integration(self.test_config)
        
        # First compilation
        result1 = integration.compile_truthgpt_model(self.test_model)
        
        # Second compilation (should use cache if available)
        result2 = integration.compile_truthgpt_model(self.test_model)
        
        self.assertIsInstance(result1, TruthGPTCompilationResult)
        self.assertIsInstance(result2, TruthGPTCompilationResult)
        
        # Both should be successful
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        
        logger.info(f"Cache test - First: {result1.primary_compiler_used}, Second: {result2.primary_compiler_used}")

class TestCompilerCore(unittest.TestCase):
    """Test suite for core compiler functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_model = TestModel()
        self.test_config = CompilationConfig(
            target=CompilationTarget.CPU,
            optimization_level=OptimizationLevel.STANDARD,
            enable_quantization=True,
            enable_fusion=True
        )
    
    def test_compiler_core_creation(self):
        """Test compiler core creation"""
        compiler = create_compiler_core(self.test_config)
        
        self.assertIsNotNone(compiler)
        self.assertEqual(compiler.config.target, CompilationTarget.CPU)
        self.assertEqual(compiler.config.optimization_level, OptimizationLevel.STANDARD)
    
    def test_compilation_result(self):
        """Test compilation result structure"""
        result = CompilationResult(success=True, compilation_time=1.0)
        
        self.assertTrue(result.success)
        self.assertEqual(result.compilation_time, 1.0)
        self.assertIsInstance(result.optimization_metrics, dict)
        self.assertIsInstance(result.warnings, list)
        self.assertIsInstance(result.errors, list)
    
    def test_compilation_config(self):
        """Test compilation configuration"""
        config = CompilationConfig(
            target=CompilationTarget.GPU,
            optimization_level=OptimizationLevel.EXTREME,
            enable_quantization=True,
            enable_fusion=True,
            enable_parallelization=True
        )
        
        self.assertEqual(config.target, CompilationTarget.GPU)
        self.assertEqual(config.optimization_level, OptimizationLevel.EXTREME)
        self.assertTrue(config.enable_quantization)
        self.assertTrue(config.enable_fusion)
        self.assertTrue(config.enable_parallelization)

class TestCompilerIntegration(unittest.TestCase):
    """Integration tests for compiler components"""
    
    def test_end_to_end_compilation(self):
        """Test end-to-end compilation workflow"""
        # Create test model
        model = TestModel()
        
        # Create TruthGPT compilation config
        config = TruthGPTCompilationConfig(
            primary_compiler="aot",
            optimization_level=OptimizationLevel.STANDARD,
            target_platform=CompilationTarget.CPU,
            enable_truthgpt_optimizations=True
        )
        
        # Create integration
        integration = create_truthgpt_compiler_integration(config)
        
        # Create mock optimizer
        optimizer = MockOptimizer("IntegrationTest")
        
        # Run compilation
        result = integration.compile_truthgpt_model(model, optimizer)
        
        # Verify result
        self.assertIsInstance(result, TruthGPTCompilationResult)
        self.assertIsNotNone(result.primary_compiler_used)
        self.assertIsInstance(result.compilation_results, dict)
        self.assertIsInstance(result.performance_metrics, dict)
        self.assertIsInstance(result.optimization_report, dict)
        
        # Verify optimizer was called
        self.assertGreater(optimizer.optimization_count, 0)
        
        logger.info(f"End-to-end test: {result.primary_compiler_used}, success: {result.success}")
    
    def test_multiple_compiler_workflow(self):
        """Test workflow with multiple compilers"""
        model = TestModel()
        
        # Test different compiler configurations
        compiler_configs = [
            ("aot", CompilationTarget.CPU),
            ("jit", CompilationTarget.CPU),
            ("mlir", CompilationTarget.CPU),
        ]
        
        results = {}
        
        for compiler_name, target in compiler_configs:
            config = TruthGPTCompilationConfig(
                primary_compiler=compiler_name,
                target_platform=target,
                optimization_level=OptimizationLevel.STANDARD
            )
            
            integration = create_truthgpt_compiler_integration(config)
            result = integration.compile_truthgpt_model(model)
            
            results[compiler_name] = result
            
            self.assertIsInstance(result, TruthGPTCompilationResult)
        
        # Verify all compilations completed
        self.assertEqual(len(results), len(compiler_configs))
        
        logger.info(f"Multiple compiler test results: {[(name, r.success) for name, r in results.items()]}")

def run_compiler_tests():
    """Run all compiler integration tests"""
    logger.info("🧪 Running TruthGPT Compiler Integration Tests")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTruthGPTCompilerIntegration,
        TestCompilerCore,
        TestCompilerIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    logger.info(f"\n📊 Test Summary:")
    logger.info(f"  Tests run: {result.testsRun}")
    logger.info(f"  Failures: {len(result.failures)}")
    logger.info(f"  Errors: {len(result.errors)}")
    logger.info(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_compiler_tests()
    exit(0 if success else 1)






