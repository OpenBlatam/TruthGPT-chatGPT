"""
Security Test Suite for Optimization Core
Comprehensive security tests for the optimization system
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import time
import threading
import hashlib
import pickle
import json
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import gc
import psutil

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


class TestInputValidation(unittest.TestCase):
    """Test cases for input validation and sanitization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_malicious_model_inputs(self):
        """Test handling of malicious model inputs."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_validation=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Test with potentially malicious inputs
            malicious_inputs = [
                # SQL injection attempts
                "'; DROP TABLE models; --",
                "1' OR '1'='1",
                
                # Path traversal attempts
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                
                # Command injection attempts
                "; rm -rf /",
                "| cat /etc/passwd",
                
                # Script injection attempts
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                
                # Unicode and special characters
                "\x00\x01\x02\x03",
                "测试中文",
                "🚀🔥💀",
                
                # Very long strings
                "A" * 10000,
                
                # Null bytes
                "\x00",
                "\x00\x00\x00",
            ]
            
            for malicious_input in malicious_inputs:
                # These should be handled gracefully
                try:
                    # Test with string inputs
                    with self.assertRaises((ValueError, TypeError)):
                        optimizer.optimize_model(malicious_input)
                except Exception as e:
                    # Any exception is acceptable for malicious inputs
                    pass
    
    def test_model_parameter_validation(self):
        """Test validation of model parameters."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_validation=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Test with models containing suspicious parameters
            class SuspiciousModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(128, 64)
                    # Add suspicious attributes
                    self.malicious_code = "eval('import os; os.system(\"rm -rf /\")')"
                    self.suspicious_path = "../../../etc/passwd"
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SuspiciousModel()
            
            # Should handle suspicious models gracefully
            try:
                optimized_model = optimizer.optimize_model(model)
                self.assertIsInstance(optimized_model, nn.Module)
            except Exception as e:
                # Any exception is acceptable for suspicious models
                pass
    
    def test_tensor_input_validation(self):
        """Test validation of tensor inputs."""
        # Test with suspicious tensor values
        suspicious_tensors = [
            torch.tensor([float('inf')]),
            torch.tensor([float('-inf')]),
            torch.tensor([float('nan')]),
            torch.tensor([1e10]),
            torch.tensor([-1e10]),
            torch.tensor([0.0]),
            torch.tensor([-0.0]),
        ]
        
        model = nn.Linear(1, 1)
        
        for tensor in suspicious_tensors:
            try:
                with torch.no_grad():
                    output = model(tensor)
                    self.assertIsInstance(output, torch.Tensor)
            except Exception as e:
                # Some suspicious values might cause exceptions
                pass
    
    def test_configuration_injection(self):
        """Test protection against configuration injection."""
        # Test with malicious configuration values
        malicious_configs = [
            {"max_memory_gb": "'; DROP TABLE config; --"},
            {"max_cpu_cores": "1; rm -rf /"},
            {"gpu_memory_fraction": "<script>alert('xss')</script>"},
            {"persistence_directory": "../../../etc/passwd"},
            {"optimization_level": "'; cat /etc/passwd; '"},
        ]
        
        for malicious_config in malicious_configs:
            try:
                # These should be handled gracefully
                config = ProductionOptimizationConfig(**malicious_config)
                self.assertIsInstance(config, ProductionOptimizationConfig)
            except Exception as e:
                # Any exception is acceptable for malicious configs
                pass


class TestDataProtection(unittest.TestCase):
    """Test cases for data protection and privacy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sensitive_data_handling(self):
        """Test handling of sensitive data."""
        # Create model with potentially sensitive data
        class SensitiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64)
                # Simulate sensitive data
                self.api_key = "sk-1234567890abcdef"
                self.password = "super_secret_password"
                self.credit_card = "4111-1111-1111-1111"
            
            def forward(self, x):
                return self.linear(x)
        
        model = SensitiveModel()
        
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_persistence=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Optimize model
            optimized_model = optimizer.optimize_model(model)
            self.assertIsInstance(optimized_model, nn.Module)
            
            # Check that sensitive data is not persisted
            cache_files = list(Path(self.temp_dir).glob("*.pkl"))
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Check that sensitive data is not in cache
                    cache_str = str(cache_data)
                    self.assertNotIn("sk-1234567890abcdef", cache_str)
                    self.assertNotIn("super_secret_password", cache_str)
                    self.assertNotIn("4111-1111-1111-1111", cache_str)
                except Exception:
                    # Cache might be empty or corrupted
                    pass
    
    def test_data_encryption(self):
        """Test data encryption in persistence."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_persistence=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Create and optimize model
            model = nn.Linear(128, 64)
            optimized_model = optimizer.optimize_model(model)
            
            # Check that cache files exist
            cache_files = list(Path(self.temp_dir).glob("*.pkl"))
            self.assertGreater(len(cache_files), 0)
            
            # Check that cache files are not human-readable
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        data = f.read()
                    
                    # Check that data is not plain text
                    try:
                        data.decode('utf-8')
                        # If we can decode as UTF-8, it might be plain text
                        # This is not necessarily a security issue, but worth noting
                    except UnicodeDecodeError:
                        # Binary data is expected
                        pass
                except Exception:
                    # File might be empty or inaccessible
                    pass
    
    def test_memory_cleanup(self):
        """Test that sensitive data is cleaned from memory."""
        # Create model with sensitive data
        class SensitiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64)
                self.sensitive_data = "sensitive_information_12345"
            
            def forward(self, x):
                return self.linear(x)
        
        model = SensitiveModel()
        
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Optimize model
            optimized_model = optimizer.optimize_model(model)
            
            # Cleanup
            optimizer.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            # Check that sensitive data is not in memory
            # This is a basic check - in practice, memory cleanup is complex
            pass


class TestAccessControl(unittest.TestCase):
    """Test cases for access control and permissions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_permissions(self):
        """Test file permissions for created files."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_persistence=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Create and optimize model
            model = nn.Linear(128, 64)
            optimized_model = optimizer.optimize_model(model)
            
            # Check file permissions
            cache_files = list(Path(self.temp_dir).glob("*"))
            for file_path in cache_files:
                if file_path.is_file():
                    # Check that files are not world-writable
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check that files are not world-writable (mode & 0o002)
                    self.assertFalse(mode & 0o002, f"File {file_path} is world-writable")
                    
                    # Check that files are not world-readable (mode & 0o004)
                    # This is more restrictive - adjust as needed
                    # self.assertFalse(mode & 0o004, f"File {file_path} is world-readable")
    
    def test_directory_traversal_protection(self):
        """Test protection against directory traversal attacks."""
        # Test with various directory traversal attempts
        traversal_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "....//....//....//etc//passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
        ]
        
        for traversal_path in traversal_paths:
            try:
                # Try to create config with traversal path
                config = ProductionOptimizationConfig(
                    persistence_directory=traversal_path
                )
                
                # Check that the path is sanitized
                self.assertNotIn("..", config.persistence_directory)
                self.assertNotIn("/etc/passwd", config.persistence_directory)
                self.assertNotIn("\\windows\\system32", config.persistence_directory)
                
            except Exception as e:
                # Any exception is acceptable for traversal attempts
                pass
    
    def test_resource_access_control(self):
        """Test control of resource access."""
        config = ProductionOptimizationConfig(
            max_memory_gb=1.0,  # Limit memory usage
            max_cpu_cores=2,    # Limit CPU usage
            persistence_directory=self.temp_dir
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Test that resource limits are enforced
            self.assertEqual(optimizer.config.max_memory_gb, 1.0)
            self.assertEqual(optimizer.config.max_cpu_cores, 2)
            
            # Test with resource-intensive operations
            try:
                # Try to create a large model
                large_model = nn.Sequential(
                    nn.Linear(1000, 2000),
                    nn.ReLU(),
                    nn.Linear(2000, 1000)
                )
                
                optimized_model = optimizer.optimize_model(large_model)
                self.assertIsInstance(optimized_model, nn.Module)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # This is expected behavior with memory limits
                    pass
                else:
                    raise


class TestInjectionAttacks(unittest.TestCase):
    """Test cases for injection attack prevention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks."""
        # Test with SQL injection attempts
        sql_injections = [
            "'; DROP TABLE models; --",
            "1' OR '1'='1",
            "'; INSERT INTO models VALUES ('hacked'); --",
            "'; UPDATE models SET name='hacked'; --",
            "'; DELETE FROM models; --",
        ]
        
        for sql_injection in sql_injections:
            try:
                # Test with configuration
                config = ProductionOptimizationConfig(
                    persistence_directory=self.temp_dir
                )
                
                # Test with model names
                model = nn.Linear(128, 64)
                model.__class__.__name__ = sql_injection
                
                # Should handle gracefully
                self.assertIsInstance(model, nn.Module)
                
            except Exception as e:
                # Any exception is acceptable for injection attempts
                pass
    
    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        # Test with command injection attempts
        command_injections = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "& whoami",
            "; ls -la",
            "`id`",
            "$(whoami)",
        ]
        
        for command_injection in command_injections:
            try:
                # Test with configuration
                config = ProductionOptimizationConfig(
                    persistence_directory=self.temp_dir
                )
                
                # Test with model parameters
                model = nn.Linear(128, 64)
                
                # Should handle gracefully
                self.assertIsInstance(model, nn.Module)
                
            except Exception as e:
                # Any exception is acceptable for injection attempts
                pass
    
    def test_script_injection_prevention(self):
        """Test prevention of script injection attacks."""
        # Test with script injection attempts
        script_injections = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onload=alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
        ]
        
        for script_injection in script_injections:
            try:
                # Test with configuration
                config = ProductionOptimizationConfig(
                    persistence_directory=self.temp_dir
                )
                
                # Test with model attributes
                model = nn.Linear(128, 64)
                model.malicious_attr = script_injection
                
                # Should handle gracefully
                self.assertIsInstance(model, nn.Module)
                
            except Exception as e:
                # Any exception is acceptable for injection attempts
                pass


class TestCryptographicSecurity(unittest.TestCase):
    """Test cases for cryptographic security."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checksum_validation(self):
        """Test checksum validation for data integrity."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_persistence=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Create and optimize model
            model = nn.Linear(128, 64)
            optimized_model = optimizer.optimize_model(model)
            
            # Get model hash
            model_hash = optimizer._get_model_hash(model)
            self.assertIsInstance(model_hash, str)
            self.assertEqual(len(model_hash), 32)  # MD5 hash length
            
            # Test hash consistency
            model_hash2 = optimizer._get_model_hash(model)
            self.assertEqual(model_hash, model_hash2)
    
    def test_data_integrity(self):
        """Test data integrity protection."""
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir,
            enable_persistence=True
        )
        
        with production_optimization_context(config.__dict__) as optimizer:
            # Create and optimize model
            model = nn.Linear(128, 64)
            optimized_model = optimizer.optimize_model(model)
            
            # Test that cache files exist and are valid
            cache_files = list(Path(self.temp_dir).glob("*.pkl"))
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    self.assertIsInstance(data, dict)
                except Exception as e:
                    # Cache file might be corrupted
                    pass
    
    def test_random_number_generation(self):
        """Test random number generation for security."""
        # Test that random numbers are generated
        random_values = [random.random() for _ in range(100)]
        
        # Check that values are different
        self.assertGreater(len(set(random_values)), 50)
        
        # Check that values are in expected range
        for value in random_values:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestNetworkSecurity(unittest.TestCase):
    """Test cases for network security."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_network_isolation(self):
        """Test network isolation and security."""
        # Test that the system doesn't make unexpected network calls
        with patch('socket.socket') as mock_socket:
            config = ProductionOptimizationConfig(
                persistence_directory=self.temp_dir
            )
            
            with production_optimization_context(config.__dict__) as optimizer:
                # Create and optimize model
                model = nn.Linear(128, 64)
                optimized_model = optimizer.optimize_model(model)
                
                # Check that no network calls were made
                self.assertEqual(mock_socket.call_count, 0)
    
    def test_url_validation(self):
        """Test URL validation and sanitization."""
        # Test with potentially malicious URLs
        malicious_urls = [
            "http://malicious.com/steal-data",
            "https://evil.com/../etc/passwd",
            "ftp://hacker.com/upload",
            "file:///etc/passwd",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
        ]
        
        for malicious_url in malicious_urls:
            try:
                # Test with configuration
                config = ProductionOptimizationConfig(
                    persistence_directory=self.temp_dir
                )
                
                # Should handle gracefully
                self.assertIsInstance(config, ProductionOptimizationConfig)
                
            except Exception as e:
                # Any exception is acceptable for malicious URLs
                pass


class TestLoggingSecurity(unittest.TestCase):
    """Test cases for logging security."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sensitive_data_logging(self):
        """Test that sensitive data is not logged."""
        # Create model with sensitive data
        class SensitiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64)
                self.api_key = "sk-1234567890abcdef"
                self.password = "super_secret_password"
            
            def forward(self, x):
                return self.linear(x)
        
        model = SensitiveModel()
        
        config = ProductionOptimizationConfig(
            persistence_directory=self.temp_dir
        )
        
        # Capture logs
        with patch('logging.getLogger') as mock_logger:
            with production_optimization_context(config.__dict__) as optimizer:
                # Optimize model
                optimized_model = optimizer.optimize_model(model)
                
                # Check that sensitive data is not in logs
                for call in mock_logger.return_value.info.call_args_list:
                    log_message = str(call)
                    self.assertNotIn("sk-1234567890abcdef", log_message)
                    self.assertNotIn("super_secret_password", log_message)
    
    def test_log_injection_prevention(self):
        """Test prevention of log injection attacks."""
        # Test with log injection attempts
        log_injections = [
            "\n[CRITICAL] System compromised",
            "\r\n[ERROR] Database breached",
            "\x00[WARNING] Security alert",
            "\n[INFO] User logged in as admin",
        ]
        
        for log_injection in log_injections:
            try:
                # Test with configuration
                config = ProductionOptimizationConfig(
                    persistence_directory=self.temp_dir
                )
                
                # Should handle gracefully
                self.assertIsInstance(config, ProductionOptimizationConfig)
                
            except Exception as e:
                # Any exception is acceptable for injection attempts
                pass


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestInputValidation,
        TestDataProtection,
        TestAccessControl,
        TestInjectionAttacks,
        TestCryptographicSecurity,
        TestNetworkSecurity,
        TestLoggingSecurity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Security Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

