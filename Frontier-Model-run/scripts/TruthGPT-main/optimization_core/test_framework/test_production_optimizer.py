"""
Test suite for Production Optimizer System
Comprehensive tests for enterprise-grade optimization with robust error handling
"""

import unittest
import tempfile
import os
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock, call
import time
import threading
from pathlib import Path

from production_optimizer import (
    ProductionOptimizer, ProductionOptimizationConfig, OptimizationLevel,
    PerformanceProfile, PerformanceMetrics, CircuitBreaker,
    create_production_optimizer, optimize_model_production,
    production_optimization_context
)


class TestProductionOptimizationConfig(unittest.TestCase):
    """Test cases for ProductionOptimizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProductionOptimizationConfig()
        
        self.assertEqual(config.optimization_level, OptimizationLevel.STANDARD)
        self.assertEqual(config.performance_profile, PerformanceProfile.BALANCED)
        self.assertEqual(config.max_memory_gb, 16.0)
        self.assertEqual(config.max_cpu_cores, 8)
        self.assertTrue(config.enable_gpu_acceleration)
        self.assertEqual(config.gpu_memory_fraction, 0.8)
        self.assertTrue(config.enable_quantization)
        self.assertTrue(config.enable_pruning)
        self.assertTrue(config.enable_kernel_fusion)
        self.assertTrue(config.enable_mixed_precision)
        self.assertTrue(config.enable_gradient_checkpointing)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = ProductionOptimizationConfig(
            max_memory_gb=32.0,
            max_cpu_cores=16,
            gpu_memory_fraction=0.9
        )
        # Should not raise any exceptions
        
        # Test invalid configurations
        with self.assertRaises(ValueError):
            ProductionOptimizationConfig(max_memory_gb=-1.0)
        
        with self.assertRaises(ValueError):
            ProductionOptimizationConfig(max_cpu_cores=0)
        
        with self.assertRaises(ValueError):
            ProductionOptimizationConfig(gpu_memory_fraction=1.5)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProductionOptimizationConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            performance_profile=PerformanceProfile.SPEED_OPTIMIZED,
            max_memory_gb=64.0,
            max_cpu_cores=32,
            enable_quantization=False,
            enable_pruning=False,
            enable_profiling=False,
            max_retry_attempts=5,
            enable_circuit_breaker=False
        )
        
        self.assertEqual(config.optimization_level, OptimizationLevel.AGGRESSIVE)
        self.assertEqual(config.performance_profile, PerformanceProfile.SPEED_OPTIMIZED)
        self.assertEqual(config.max_memory_gb, 64.0)
        self.assertEqual(config.max_cpu_cores, 32)
        self.assertFalse(config.enable_quantization)
        self.assertFalse(config.enable_pruning)
        self.assertFalse(config.enable_profiling)
        self.assertEqual(config.max_retry_attempts, 5)
        self.assertFalse(config.enable_circuit_breaker)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = PerformanceMetrics()
    
    def test_timer_operations(self):
        """Test timer start and end operations."""
        # Start timer
        timer_id = self.metrics.start_timer("test_operation")
        self.assertIsInstance(timer_id, str)
        self.assertIn("test_operation", timer_id)
        
        # Wait a small amount
        time.sleep(0.01)
        
        # End timer
        duration = self.metrics.end_timer(timer_id, "test_operation")
        self.assertIsInstance(duration, float)
        self.assertGreater(duration, 0)
        
        # Check metrics were recorded
        summary = self.metrics.get_metrics_summary()
        self.assertIn("test_operation_duration", summary)
    
    def test_metric_recording(self):
        """Test custom metric recording."""
        self.metrics.record_metric("custom_metric", 42.0)
        self.metrics.record_metric("custom_metric", 24.0)
        
        summary = self.metrics.get_metrics_summary()
        self.assertIn("custom_metric", summary)
        
        metric_data = summary["custom_metric"]
        self.assertEqual(metric_data["count"], 2)
        self.assertEqual(metric_data["mean"], 33.0)
        self.assertEqual(metric_data["min"], 24.0)
        self.assertEqual(metric_data["max"], 42.0)
        self.assertEqual(metric_data["latest"], 24.0)
    
    def test_thread_safety(self):
        """Test thread safety of metrics collection."""
        results = []
        
        def worker(thread_id):
            for i in range(10):
                timer_id = self.metrics.start_timer(f"thread_{thread_id}")
                time.sleep(0.001)
                self.metrics.end_timer(timer_id, f"thread_{thread_id}")
                self.metrics.record_metric(f"thread_{thread_id}_metric", i)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete without errors
        summary = self.metrics.get_metrics_summary()
        self.assertGreater(len(summary), 0)


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for CircuitBreaker class."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
        
        # Should allow calls in CLOSED state
        result = breaker.call(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(breaker.state, "CLOSED")
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker failure threshold."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        # First failure
        with self.assertRaises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        self.assertEqual(breaker.failure_count, 1)
        self.assertEqual(breaker.state, "CLOSED")
        
        # Second failure - should open circuit
        with self.assertRaises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        self.assertEqual(breaker.failure_count, 2)
        self.assertEqual(breaker.state, "OPEN")
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in OPEN state."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=1.0)
        
        # Trigger failure to open circuit
        with self.assertRaises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        self.assertEqual(breaker.state, "OPEN")
        
        # Should raise exception when circuit is open
        with self.assertRaises(Exception, msg="Circuit breaker is OPEN"):
            breaker.call(lambda: "should not execute")
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery from OPEN to CLOSED."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)
        
        # Trigger failure to open circuit
        with self.assertRaises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        self.assertEqual(breaker.state, "OPEN")
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Successful call should close circuit
        result = breaker.call(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(breaker.state, "CLOSED")
        self.assertEqual(breaker.failure_count, 0)


class TestProductionOptimizer(unittest.TestCase):
    """Test cases for ProductionOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_persistence=True,
            enable_result_caching=True
        )
        self.optimizer = ProductionOptimizer(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.optimizer.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.config, self.config)
        self.assertIsInstance(self.optimizer.metrics, PerformanceMetrics)
        self.assertIsInstance(self.optimizer.optimization_cache, dict)
        self.assertIsInstance(self.optimizer.operation_history, list)
    
    def test_simple_model_optimization(self):
        """Test optimization of a simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        original_params = sum(p.numel() for p in model.parameters())
        
        # Optimize model
        optimized_model = self.optimizer.optimize_model(model)
        
        # Should return a model
        self.assertIsInstance(optimized_model, nn.Module)
        
        # Test that model can still forward pass
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            output = optimized_model(test_input)
            self.assertEqual(output.shape, (1, 5))
    
    def test_optimization_caching(self):
        """Test optimization result caching."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 3)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # First optimization
        optimized1 = self.optimizer.optimize_model(model)
        
        # Second optimization should use cache
        optimized2 = self.optimizer.optimize_model(model)
        
        # Should be the same object (cached)
        self.assertIs(optimized1, optimized2)
    
    def test_optimization_strategies(self):
        """Test different optimization strategies."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.conv = nn.Conv2d(1, 3, 3)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test with different optimization levels
        configs = [
            OptimizationLevel.MINIMAL,
            OptimizationLevel.STANDARD,
            OptimizationLevel.AGGRESSIVE,
            OptimizationLevel.MAXIMUM
        ]
        
        for level in configs:
            with self.subTest(level=level):
                config = ProductionOptimizationConfig(
                    optimization_level=level,
                    persistence_directory=self.temp_dir
                )
                optimizer = ProductionOptimizer(config)
                
                try:
                    optimized = optimizer.optimize_model(model)
                    self.assertIsInstance(optimized, nn.Module)
                finally:
                    optimizer.cleanup()
    
    def test_performance_profiles(self):
        """Test different performance profiles."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        profiles = [
            PerformanceProfile.MEMORY_OPTIMIZED,
            PerformanceProfile.SPEED_OPTIMIZED,
            PerformanceProfile.BALANCED,
            PerformanceProfile.CUSTOM
        ]
        
        for profile in profiles:
            with self.subTest(profile=profile):
                config = ProductionOptimizationConfig(
                    performance_profile=profile,
                    persistence_directory=self.temp_dir
                )
                optimizer = ProductionOptimizer(config)
                
                try:
                    optimized = optimizer.optimize_model(model)
                    self.assertIsInstance(optimized, nn.Module)
                finally:
                    optimizer.cleanup()
    
    def test_optimization_validation(self):
        """Test optimization validation."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test with validation enabled
        config = ProductionOptimizationConfig(
            enable_validation=True,
            persistence_directory=self.temp_dir
        )
        optimizer = ProductionOptimizer(config)
        
        try:
            optimized = optimizer.optimize_model(model)
            self.assertIsInstance(optimized, nn.Module)
        finally:
            optimizer.cleanup()
    
    def test_error_handling(self):
        """Test error handling in optimization."""
        # Test with invalid model
        with self.assertRaises(ValueError):
            self.optimizer.optimize_model("not a model")
        
        # Test with None model
        with self.assertRaises(ValueError):
            self.optimizer.optimize_model(None)
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Optimize model to generate metrics
        self.optimizer.optimize_model(model)
        
        # Get performance metrics
        metrics = self.optimizer.get_performance_metrics()
        
        self.assertIn('optimization_metrics', metrics)
        self.assertIn('system_metrics', metrics)
        self.assertIn('cache_metrics', metrics)
        
        # Check that metrics were recorded
        optimization_metrics = metrics['optimization_metrics']
        self.assertIsInstance(optimization_metrics, dict)
    
    def test_cleanup(self):
        """Test optimizer cleanup."""
        # Create some data
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        self.optimizer.optimize_model(model)
        
        # Cleanup should not raise exceptions
        self.optimizer.cleanup()
    
    @patch('torch.cuda.is_available')
    def test_gpu_memory_usage(self, mock_cuda_available):
        """Test GPU memory usage calculation."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.memory_allocated', return_value=1024**3):  # 1GB
            gpu_memory = self.optimizer._get_gpu_memory_usage()
            self.assertEqual(gpu_memory, 1.0)
        
        mock_cuda_available.return_value = False
        gpu_memory = self.optimizer._get_gpu_memory_usage()
        self.assertEqual(gpu_memory, 0.0)
    
    def test_model_hash_generation(self):
        """Test model hash generation for caching."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model1 = TestModel()
        model2 = TestModel()
        
        hash1 = self.optimizer._get_model_hash(model1)
        hash2 = self.optimizer._get_model_hash(model2)
        
        # Different model instances should have different hashes
        self.assertNotEqual(hash1, hash2)
        
        # Same model should have same hash
        hash1_again = self.optimizer._get_model_hash(model1)
        self.assertEqual(hash1, hash1_again)
        
        # Hash should be a string
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 32)  # MD5 hash length


class TestFactoryFunctions(unittest.TestCase):
    """Test cases for factory functions."""
    
    def test_create_production_optimizer(self):
        """Test create_production_optimizer factory function."""
        config_dict = {
            'optimization_level': OptimizationLevel.AGGRESSIVE,
            'max_memory_gb': 32.0,
            'enable_quantization': False
        }
        
        optimizer = create_production_optimizer(config_dict)
        
        self.assertIsInstance(optimizer, ProductionOptimizer)
        self.assertEqual(optimizer.config.optimization_level, OptimizationLevel.AGGRESSIVE)
        self.assertEqual(optimizer.config.max_memory_gb, 32.0)
        self.assertFalse(optimizer.config.enable_quantization)
    
    def test_create_production_optimizer_default(self):
        """Test create_production_optimizer with default config."""
        optimizer = create_production_optimizer()
        
        self.assertIsInstance(optimizer, ProductionOptimizer)
        self.assertEqual(optimizer.config.optimization_level, OptimizationLevel.STANDARD)
    
    def test_optimize_model_production(self):
        """Test optimize_model_production function."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        config = {'optimization_level': OptimizationLevel.STANDARD}
        
        optimized = optimize_model_production(model, config)
        
        self.assertIsInstance(optimized, nn.Module)
    
    def test_production_optimization_context(self):
        """Test production_optimization_context context manager."""
        config = {'optimization_level': OptimizationLevel.AGGRESSIVE}
        
        with production_optimization_context(config) as optimizer:
            self.assertIsInstance(optimizer, ProductionOptimizer)
            self.assertEqual(optimizer.config.optimization_level, OptimizationLevel.AGGRESSIVE)
        
        # Optimizer should be cleaned up after context exit


class TestOptimizationStrategies(unittest.TestCase):
    """Test cases for individual optimization strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_persistence=False  # Disable for faster tests
        )
        self.optimizer = ProductionOptimizer(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.optimizer.cleanup()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_quantization_strategy(self):
        """Test quantization strategy."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.conv = nn.Conv2d(1, 3, 3)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test with quantization enabled
        config = ProductionOptimizationConfig(
            enable_quantization=True,
            optimization_level=OptimizationLevel.STANDARD,
            persistence_directory=self.temp_dir
        )
        optimizer = ProductionOptimizer(config)
        
        try:
            optimized = optimizer.optimize_model(model)
            self.assertIsInstance(optimized, nn.Module)
        finally:
            optimizer.cleanup()
    
    def test_pruning_strategy(self):
        """Test pruning strategy."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.conv = nn.Conv2d(1, 3, 3)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test with pruning enabled
        config = ProductionOptimizationConfig(
            enable_pruning=True,
            optimization_level=OptimizationLevel.STANDARD,
            persistence_directory=self.temp_dir
        )
        optimizer = ProductionOptimizer(config)
        
        try:
            optimized = optimizer.optimize_model(model)
            self.assertIsInstance(optimized, nn.Module)
        finally:
            optimizer.cleanup()
    
    def test_mixed_precision_strategy(self):
        """Test mixed precision strategy."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test with mixed precision enabled
        config = ProductionOptimizationConfig(
            enable_mixed_precision=True,
            performance_profile=PerformanceProfile.MEMORY_OPTIMIZED,
            persistence_directory=self.temp_dir
        )
        optimizer = ProductionOptimizer(config)
        
        try:
            optimized = optimizer.optimize_model(model)
            self.assertIsInstance(optimized, nn.Module)
        finally:
            optimizer.cleanup()
    
    def test_gradient_checkpointing_strategy(self):
        """Test gradient checkpointing strategy."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Test with gradient checkpointing enabled
        config = ProductionOptimizationConfig(
            enable_gradient_checkpointing=True,
            persistence_directory=self.temp_dir
        )
        optimizer = ProductionOptimizer(config)
        
        try:
            optimized = optimizer.optimize_model(model)
            self.assertIsInstance(optimized, nn.Module)
        finally:
            optimizer.cleanup()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete optimization system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization workflow."""
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(20, 50)
                self.linear2 = nn.Linear(50, 30)
                self.linear3 = nn.Linear(30, 10)
                self.conv = nn.Conv2d(1, 16, 3)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = torch.relu(x)
                x = self.linear3(x)
                return x
        
        model = ComplexModel()
        
        # Test with comprehensive configuration
        config = ProductionOptimizationConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            performance_profile=PerformanceProfile.SPEED_OPTIMIZED,
            enable_quantization=True,
            enable_pruning=True,
            enable_kernel_fusion=True,
            enable_mixed_precision=True,
            enable_gradient_checkpointing=True,
            enable_profiling=True,
            enable_validation=True,
            enable_result_caching=True,
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config) as optimizer:
            # Optimize model
            optimized_model = optimizer.optimize_model(model)
            
            # Verify optimization worked
            self.assertIsInstance(optimized_model, nn.Module)
            
            # Test forward pass
            test_input = torch.randn(1, 20)
            with torch.no_grad():
                output = optimized_model(test_input)
                self.assertEqual(output.shape, (1, 10))
            
            # Get performance metrics
            metrics = optimizer.get_performance_metrics()
            self.assertIn('optimization_metrics', metrics)
            self.assertIn('system_metrics', metrics)
            self.assertIn('cache_metrics', metrics)
    
    def test_multiple_model_optimization(self):
        """Test optimization of multiple different models."""
        models = []
        
        # Create different model architectures
        for i in range(3):
            class TestModel(nn.Module):
                def __init__(self, size):
                    super().__init__()
                    self.linear = nn.Linear(size, size // 2)
                
                def forward(self, x):
                    return self.linear(x)
            
            models.append(TestModel(10 + i * 5))
        
        config = ProductionOptimizationConfig(
            enable_result_caching=True,
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config) as optimizer:
            optimized_models = []
            
            for model in models:
                optimized = optimizer.optimize_model(model)
                optimized_models.append(optimized)
                self.assertIsInstance(optimized, nn.Module)
            
            # Test that caching worked
            for i, model in enumerate(models):
                optimized_again = optimizer.optimize_model(model)
                self.assertIs(optimized_models[i], optimized_again)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestProductionOptimizationConfig,
        TestPerformanceMetrics,
        TestCircuitBreaker,
        TestProductionOptimizer,
        TestFactoryFunctions,
        TestOptimizationStrategies,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

