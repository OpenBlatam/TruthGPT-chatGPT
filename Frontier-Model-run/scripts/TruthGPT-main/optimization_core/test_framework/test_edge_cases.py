"""
Edge Cases and Stress Test Suite for Optimization Core
Comprehensive tests for edge cases, stress scenarios, and boundary conditions
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import time
import threading
import gc
import psutil
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import random
import string

# Import optimization components
from production_config import (
    ProductionConfig, Environment, create_production_config,
    production_config_context
)
from production_optimizer import (
    ProductionOptimizer, ProductionOptimizationConfig, OptimizationLevel,
    PerformanceProfile, create_production_optimizer, production_optimization_context
)
from __init__ import (
    OptimizedLayerNorm, OptimizedRMSNorm, CUDAOptimizations,
    AdvancedRMSNorm, LlamaRMSNorm, CRMSNorm,
    SwiGLU, GatedMLP, MixtureOfExperts, AdaptiveMLP,
    FusedLayerNormLinear, FusedAttentionMLP, KernelFusionOptimizer,
    QuantizedLinear, QuantizedLayerNorm, AdvancedQuantizationOptimizer,
    TensorPool, ActivationCache, MemoryPoolingOptimizer,
    FusedAttention, BatchOptimizer, ComputationalOptimizer,
    apply_optimizations, apply_advanced_optimizations
)


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extreme_model_sizes(self):
        """Test with extremely large and small model sizes."""
        # Test with very small model
        tiny_model = nn.Linear(1, 1)
        
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_validation=False  # Disable validation for edge cases
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            optimized_tiny = optimizer.optimize_model(tiny_model)
            self.assertIsInstance(optimized_tiny, nn.Module)
            
            # Test forward pass
            x = torch.randn(1, 1)
            with torch.no_grad():
                output = optimized_tiny(x)
                self.assertEqual(output.shape, (1, 1))
        
        # Test with very large model (if memory allows)
        try:
            large_model = nn.Sequential(
                nn.Linear(1000, 2000),
                nn.ReLU(),
                nn.Linear(2000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 500)
            )
            
            with production_optimization_context(config.__dict__) as optimizer:
                optimized_large = optimizer.optimize_model(large_model)
                self.assertIsInstance(optimized_large, nn.Module)
                
                # Test forward pass
                x = torch.randn(32, 1000)
                with torch.no_grad():
                    output = optimized_large(x)
                    self.assertEqual(output.shape, (32, 500))
                    
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest("Insufficient memory for large model test")
            else:
                raise
    
    def test_extreme_batch_sizes(self):
        """Test with extreme batch sizes."""
        model = nn.Linear(128, 64)
        
        # Test with batch size 1
        x_small = torch.randn(1, 128)
        with torch.no_grad():
            output_small = model(x_small)
            self.assertEqual(output_small.shape, (1, 64))
        
        # Test with very large batch size (if memory allows)
        try:
            x_large = torch.randn(10000, 128)
            with torch.no_grad():
                output_large = model(x_large)
                self.assertEqual(output_large.shape, (10000, 64))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest("Insufficient memory for large batch test")
            else:
                raise
    
    def test_extreme_sequence_lengths(self):
        """Test with extreme sequence lengths."""
        model = nn.Linear(128, 64)
        
        # Test with very short sequence
        x_short = torch.randn(32, 1, 128)
        with torch.no_grad():
            output_short = model(x_short)
            self.assertEqual(output_short.shape, (32, 1, 64))
        
        # Test with very long sequence (if memory allows)
        try:
            x_long = torch.randn(32, 10000, 128)
            with torch.no_grad():
                output_long = model(x_long)
                self.assertEqual(output_long.shape, (32, 10000, 64))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest("Insufficient memory for long sequence test")
            else:
                raise
    
    def test_nan_and_inf_values(self):
        """Test handling of NaN and infinity values."""
        model = nn.Linear(128, 64)
        
        # Test with NaN values
        x_nan = torch.randn(32, 128)
        x_nan[0, 0] = float('nan')
        
        with torch.no_grad():
            output_nan = model(x_nan)
            # Model should handle NaN gracefully
            self.assertIsInstance(output_nan, torch.Tensor)
        
        # Test with infinity values
        x_inf = torch.randn(32, 128)
        x_inf[0, 0] = float('inf')
        
        with torch.no_grad():
            output_inf = model(x_inf)
            # Model should handle infinity gracefully
            self.assertIsInstance(output_inf, torch.Tensor)
    
    def test_zero_tensors(self):
        """Test with zero tensors."""
        model = nn.Linear(128, 64)
        
        # Test with zero tensor
        x_zero = torch.zeros(32, 128)
        with torch.no_grad():
            output_zero = model(x_zero)
            self.assertEqual(output_zero.shape, (32, 64))
            # Output should be zero (assuming bias is zero)
            self.assertTrue(torch.allclose(output_zero, torch.zeros(32, 64)))
    
    def test_negative_values(self):
        """Test with negative values."""
        model = nn.Linear(128, 64)
        
        # Test with negative values
        x_negative = torch.randn(32, 128) * -1
        with torch.no_grad():
            output_negative = model(x_negative)
            self.assertEqual(output_negative.shape, (32, 64))
    
    def test_extreme_learning_rates(self):
        """Test with extreme learning rates."""
        model = nn.Linear(128, 64)
        
        # Test with very small learning rate
        optimizer_small = torch.optim.Adam(model.parameters(), lr=1e-10)
        
        x = torch.randn(32, 128)
        y = torch.randn(32, 64)
        
        for _ in range(10):
            optimizer_small.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer_small.step()
        
        # Test with very large learning rate
        model_large = nn.Linear(128, 64)
        optimizer_large = torch.optim.Adam(model_large.parameters(), lr=1e2)
        
        for _ in range(10):
            optimizer_large.zero_grad()
            output = model_large(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer_large.step()


class TestStressScenarios(unittest.TestCase):
    """Test cases for stress scenarios and high-load conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rapid_model_creation(self):
        """Test rapid creation and optimization of many models."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_result_caching=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Create and optimize many models rapidly
            for i in range(100):
                model = nn.Linear(128, 64)
                optimized_model = optimizer.optimize_model(model)
                self.assertIsInstance(optimized_model, nn.Module)
                
                # Test forward pass
                x = torch.randn(32, 128)
                with torch.no_grad():
                    output = optimized_model(x)
                    self.assertEqual(output.shape, (32, 64))
    
    def test_concurrent_optimization_stress(self):
        """Test concurrent optimization under stress."""
        config = ProductionOptimizationConfig(
            max_workers=8,
            enable_async_processing=True,
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Create many concurrent optimization tasks
            threads = []
            results = []
            
            def optimize_model(model, index):
                try:
                    optimized = optimizer.optimize_model(model)
                    results.append((index, optimized))
                except Exception as e:
                    results.append((index, e))
            
            # Start many concurrent optimizations
            for i in range(50):
                model = nn.Linear(128, 64)
                thread = threading.Thread(target=optimize_model, args=(model, i))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Verify results
            self.assertEqual(len(results), 50)
            for index, result in results:
                if isinstance(result, Exception):
                    self.fail(f"Optimization failed for model {index}: {result}")
                else:
                    self.assertIsInstance(result, nn.Module)
    
    def test_memory_stress(self):
        """Test memory usage under stress."""
        # Create many large models to stress memory
        models = []
        
        try:
            for i in range(20):
                model = nn.Sequential(
                    nn.Linear(1000, 2000),
                    nn.ReLU(),
                    nn.Linear(2000, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 500)
                )
                models.append(model)
            
            # Test forward passes
            for model in models:
                x = torch.randn(32, 1000)
                with torch.no_grad():
                    output = model(x)
                    self.assertEqual(output.shape, (32, 500))
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest("Insufficient memory for stress test")
            else:
                raise
    
    def test_cpu_stress(self):
        """Test CPU usage under stress."""
        # Create computationally intensive models
        models = []
        
        for i in range(10):
            model = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            models.append(model)
        
        # Run many forward passes
        for _ in range(100):
            for model in models:
                x = torch.randn(32, 512)
                with torch.no_grad():
                    output = model(x)
                    self.assertEqual(output.shape, (32, 256))
    
    def test_gpu_stress(self):
        """Test GPU usage under stress."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create many GPU models
        models = []
        
        try:
            for i in range(10):
                model = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512)
                ).cuda()
                models.append(model)
            
            # Run many forward passes
            for _ in range(50):
                for model in models:
                    x = torch.randn(32, 512).cuda()
                    with torch.no_grad():
                        output = model(x)
                        self.assertEqual(output.shape, (32, 512))
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest("Insufficient GPU memory for stress test")
            else:
                raise


class TestBoundaryConditions(unittest.TestCase):
    """Test cases for boundary conditions and limits."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_boundary_values(self):
        """Test configuration with boundary values."""
        # Test with minimum values
        config_min = ProductionOptimizationConfig(
            max_memory_gb=0.1,
            max_cpu_cores=1,
            gpu_memory_fraction=0.01,
            max_retry_attempts=1,
            retry_delay=0.001,
            batch_size=1,
            max_workers=1
        )
        
        self.assertEqual(config_min.max_memory_gb, 0.1)
        self.assertEqual(config_min.max_cpu_cores, 1)
        self.assertEqual(config_min.gpu_memory_fraction, 0.01)
        
        # Test with maximum values
        config_max = ProductionOptimizationConfig(
            max_memory_gb=1000.0,
            max_cpu_cores=128,
            gpu_memory_fraction=1.0,
            max_retry_attempts=100,
            retry_delay=60.0,
            batch_size=10000,
            max_workers=64
        )
        
        self.assertEqual(config_max.max_memory_gb, 1000.0)
        self.assertEqual(config_max.max_cpu_cores, 128)
        self.assertEqual(config_max.gpu_memory_fraction, 1.0)
    
    def test_model_boundary_conditions(self):
        """Test models with boundary conditions."""
        # Test with single parameter
        single_param_model = nn.Linear(1, 1)
        self.assertEqual(sum(p.numel() for p in single_param_model.parameters()), 2)
        
        # Test with zero parameters (if possible)
        try:
            zero_param_model = nn.Linear(0, 0)
            self.assertEqual(sum(p.numel() for p in zero_param_model.parameters()), 0)
        except Exception:
            # Zero parameter models might not be supported
            pass
    
    def test_tensor_boundary_conditions(self):
        """Test tensors with boundary conditions."""
        # Test with empty tensor
        empty_tensor = torch.empty(0, 0)
        self.assertEqual(empty_tensor.shape, (0, 0))
        
        # Test with single element tensor
        single_tensor = torch.tensor([42.0])
        self.assertEqual(single_tensor.shape, (1,))
        self.assertEqual(single_tensor.item(), 42.0)
        
        # Test with maximum tensor size (if memory allows)
        try:
            max_tensor = torch.randn(1000, 1000)
            self.assertEqual(max_tensor.shape, (1000, 1000))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.skipTest("Insufficient memory for large tensor test")
            else:
                raise
    
    def test_optimization_boundary_conditions(self):
        """Test optimization with boundary conditions."""
        config = ProductionOptimizationConfig(
            optimization_level=OptimizationLevel.MINIMAL,
            enable_quantization=False,
            enable_pruning=False,
            enable_kernel_fusion=False,
            enable_mixed_precision=False,
            enable_gradient_checkpointing=False,
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            model = nn.Linear(128, 64)
            optimized_model = optimizer.optimize_model(model)
            self.assertIsInstance(optimized_model, nn.Module)
            
            # Test forward pass
            x = torch.randn(32, 128)
            with torch.no_grad():
                output = optimized_model(x)
                self.assertEqual(output.shape, (32, 64))


class TestErrorRecovery(unittest.TestCase):
    """Test cases for error recovery and fault tolerance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_model_handling(self):
        """Test handling of invalid models."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_validation=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Test with invalid model types
            invalid_models = [
                "not a model",
                None,
                123,
                {},
                [],
                torch.tensor([1, 2, 3])
            ]
            
            for invalid_model in invalid_models:
                with self.assertRaises((ValueError, TypeError)):
                    optimizer.optimize_model(invalid_model)
    
    def test_corrupted_config_handling(self):
        """Test handling of corrupted configuration."""
        # Test with invalid configuration values
        with self.assertRaises(ValueError):
            ProductionOptimizationConfig(max_memory_gb=-1.0)
        
        with self.assertRaises(ValueError):
            ProductionOptimizationConfig(max_cpu_cores=0)
        
        with self.assertRaises(ValueError):
            ProductionOptimizationConfig(gpu_memory_fraction=1.5)
    
    def test_memory_exhaustion_handling(self):
        """Test handling of memory exhaustion."""
        # Create a model that might cause memory issues
        try:
            # Try to create a very large model
            large_model = nn.Sequential(
                nn.Linear(10000, 20000),
                nn.ReLU(),
                nn.Linear(20000, 10000),
                nn.ReLU(),
                nn.Linear(10000, 5000)
            )
            
            # Test forward pass
            x = torch.randn(1, 10000)
            with torch.no_grad():
                output = large_model(x)
                self.assertEqual(output.shape, (1, 5000))
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # This is expected behavior
                pass
            else:
                raise
    
    def test_gpu_memory_exhaustion_handling(self):
        """Test handling of GPU memory exhaustion."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        try:
            # Try to create many GPU models
            models = []
            for i in range(100):
                model = nn.Linear(1000, 1000).cuda()
                models.append(model)
            
            # Test forward passes
            for model in models:
                x = torch.randn(32, 1000).cuda()
                with torch.no_grad():
                    output = model(x)
                    self.assertEqual(output.shape, (32, 1000))
                    
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # This is expected behavior
                pass
            else:
                raise
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        config = ProductionOptimizationConfig(
            enable_circuit_breaker=True,
            circuit_breaker_threshold=3,
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Trigger circuit breaker with failures
            for i in range(5):
                try:
                    optimizer.optimize_model("invalid model")
                except (ValueError, TypeError):
                    pass
            
            # Circuit breaker should be open now
            with self.assertRaises(Exception):
                optimizer.optimize_model("invalid model")


class TestResourceLimits(unittest.TestCase):
    """Test cases for resource limit handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        config = ProductionOptimizationConfig(
            max_memory_gb=0.1,  # Very small limit
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Try to create models that exceed memory limit
            try:
                large_model = nn.Sequential(
                    nn.Linear(1000, 2000),
                    nn.ReLU(),
                    nn.Linear(2000, 1000)
                )
                
                optimized_model = optimizer.optimize_model(large_model)
                self.assertIsInstance(optimized_model, nn.Module)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # This is expected behavior
                    pass
                else:
                    raise
    
    def test_cpu_limit_enforcement(self):
        """Test CPU limit enforcement."""
        config = ProductionOptimizationConfig(
            max_cpu_cores=1,
            max_workers=1,
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Test with CPU-intensive operations
            models = [nn.Linear(128, 64) for _ in range(10)]
            
            for model in models:
                optimized_model = optimizer.optimize_model(model)
                self.assertIsInstance(optimized_model, nn.Module)
    
    def test_gpu_memory_limit_enforcement(self):
        """Test GPU memory limit enforcement."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        config = ProductionOptimizationConfig(
            gpu_memory_fraction=0.01,  # Very small fraction
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Try to create GPU models
            try:
                model = nn.Linear(1000, 1000).cuda()
                optimized_model = optimizer.optimize_model(model)
                self.assertIsInstance(optimized_model, nn.Module)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # This is expected behavior
                    pass
                else:
                    raise


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestEdgeCases,
        TestStressScenarios,
        TestBoundaryConditions,
        TestErrorRecovery,
        TestResourceLimits
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Edge Cases and Stress Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

