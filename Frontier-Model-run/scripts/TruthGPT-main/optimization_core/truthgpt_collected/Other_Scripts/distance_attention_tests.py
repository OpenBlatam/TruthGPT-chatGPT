#!/usr/bin/env python3
"""
Comprehensive Testing and Validation for Distance-Based Attention
================================================================

This module provides comprehensive testing, validation, and benchmarking
for the distance-based attention mechanism implementation.

Features:
- Unit tests for all distance metrics
- Performance benchmarking
- Memory usage analysis
- Gradient flow validation
- Numerical stability tests
- Integration tests with TruthGPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
import unittest
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our modules
from distance_based_attention import (
    DistanceBasedAttention,
    MultiHeadDistanceAttention,
    DistanceAttentionConfig,
    DistanceType,
    create_distance_attention
)
from truthgpt_distance_attention_integration import (
    TruthGPTDistanceAttentionModel,
    TruthGPTAttentionConfig,
    DistanceAttentionOptimizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for testing parameters."""
    batch_size: int = 4
    seq_len: int = 128
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 6
    vocab_size: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

class DistanceAttentionTester:
    """
    Comprehensive tester for distance-based attention mechanisms.
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = {}
        self.device = torch.device(config.device)
        
        logger.info(f"Initialized tester with device: {self.device}")
    
    def test_distance_metrics(self) -> Dict[str, Any]:
        """Test all distance metrics for correctness."""
        logger.info("Testing distance metrics...")
        
        results = {}
        
        # Create test data
        batch_size, num_heads, seq_len, head_dim = 2, 4, 16, 64
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        
        # Test each distance type
        for distance_type in DistanceType:
            logger.info(f"Testing {distance_type.value} distance...")
            
            config = DistanceAttentionConfig(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                distance_type=distance_type.value,
                lambda_param=1.0
            )
            
            attention = create_distance_attention(config).to(self.device)
            
            # Test forward pass
            try:
                with torch.no_grad():
                    output, attn_weights = attention(Q.view(batch_size, seq_len, -1), output_attentions=True)
                
                # Validate output shapes
                assert output.shape == (batch_size, seq_len, self.config.hidden_size), \
                    f"Output shape mismatch for {distance_type.value}"
                assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
                    f"Attention weights shape mismatch for {distance_type.value}"
                
                # Validate attention weights sum to 1
                attn_sum = attn_weights.sum(dim=-1)
                assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6), \
                    f"Attention weights don't sum to 1 for {distance_type.value}"
                
                # Validate attention weights are non-negative
                assert torch.all(attn_weights >= 0), \
                    f"Negative attention weights found for {distance_type.value}"
                
                results[distance_type.value] = {
                    'status': 'PASSED',
                    'output_shape': output.shape,
                    'attention_shape': attn_weights.shape,
                    'attention_sum_mean': attn_sum.mean().item(),
                    'attention_sum_std': attn_sum.std().item(),
                    'attention_min': attn_weights.min().item(),
                    'attention_max': attn_weights.max().item()
                }
                
                logger.info(f"✓ {distance_type.value} distance test passed")
                
            except Exception as e:
                results[distance_type.value] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                logger.error(f"✗ {distance_type.value} distance test failed: {e}")
        
        self.results['distance_metrics'] = results
        return results
    
    def test_gradient_flow(self) -> Dict[str, Any]:
        """Test gradient flow through the attention mechanism."""
        logger.info("Testing gradient flow...")
        
        results = {}
        
        # Create test data with requires_grad
        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, self.config.hidden_size, 
                       device=self.device, requires_grad=True)
        
        for distance_type in [DistanceType.L1, DistanceType.L2, DistanceType.LP]:
            logger.info(f"Testing gradient flow for {distance_type.value}...")
            
            config = DistanceAttentionConfig(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                distance_type=distance_type.value,
                lambda_param=1.0
            )
            
            attention = create_distance_attention(config).to(self.device)
            
            try:
                # Forward pass
                output, _ = attention(x, output_attentions=False)
                
                # Compute loss
                loss = output.sum()
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                has_gradients = x.grad is not None and torch.any(x.grad != 0)
                gradient_norm = x.grad.norm().item() if x.grad is not None else 0.0
                
                # Check for NaN or Inf gradients
                has_nan_grad = torch.isnan(x.grad).any() if x.grad is not None else False
                has_inf_grad = torch.isinf(x.grad).any() if x.grad is not None else False
                
                results[distance_type.value] = {
                    'status': 'PASSED' if has_gradients and not has_nan_grad and not has_inf_grad else 'FAILED',
                    'has_gradients': has_gradients,
                    'gradient_norm': gradient_norm,
                    'has_nan_gradients': has_nan_grad,
                    'has_inf_gradients': has_inf_grad
                }
                
                if has_gradients and not has_nan_grad and not has_inf_grad:
                    logger.info(f"✓ Gradient flow test passed for {distance_type.value}")
                else:
                    logger.error(f"✗ Gradient flow test failed for {distance_type.value}")
                
            except Exception as e:
                results[distance_type.value] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                logger.error(f"✗ Gradient flow test failed for {distance_type.value}: {e}")
        
        self.results['gradient_flow'] = results
        return results
    
    def test_numerical_stability(self) -> Dict[str, Any]:
        """Test numerical stability with extreme values."""
        logger.info("Testing numerical stability...")
        
        results = {}
        
        # Test with extreme values
        test_cases = [
            ('normal', torch.randn(2, 16, self.config.hidden_size)),
            ('large_values', torch.randn(2, 16, self.config.hidden_size) * 100),
            ('small_values', torch.randn(2, 16, self.config.hidden_size) * 0.001),
            ('mixed_signs', torch.randn(2, 16, self.config.hidden_size) * 10),
        ]
        
        for case_name, x in test_cases:
            x = x.to(self.device)
            results[case_name] = {}
            
            for distance_type in [DistanceType.L1, DistanceType.L2]:
                logger.info(f"Testing {case_name} with {distance_type.value}...")
                
                config = DistanceAttentionConfig(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    distance_type=distance_type.value,
                    lambda_param=1.0
                )
                
                attention = create_distance_attention(config).to(self.device)
                
                try:
                    with torch.no_grad():
                        output, attn_weights = attention(x, output_attentions=True)
                    
                    # Check for NaN or Inf
                    has_nan_output = torch.isnan(output).any()
                    has_inf_output = torch.isinf(output).any()
                    has_nan_attn = torch.isnan(attn_weights).any()
                    has_inf_attn = torch.isinf(attn_weights).any()
                    
                    # Check attention weights are valid probabilities
                    attn_sum = attn_weights.sum(dim=-1)
                    valid_probs = torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)
                    
                    results[case_name][distance_type.value] = {
                        'status': 'PASSED' if not (has_nan_output or has_inf_output or has_nan_attn or has_inf_attn) and valid_probs else 'FAILED',
                        'has_nan_output': has_nan_output,
                        'has_inf_output': has_inf_output,
                        'has_nan_attention': has_nan_attn,
                        'has_inf_attention': has_inf_attn,
                        'valid_probabilities': valid_probs,
                        'output_range': [output.min().item(), output.max().item()],
                        'attention_range': [attn_weights.min().item(), attn_weights.max().item()]
                    }
                    
                    if not (has_nan_output or has_inf_output or has_nan_attn or has_inf_attn) and valid_probs:
                        logger.info(f"✓ Numerical stability test passed for {case_name} with {distance_type.value}")
                    else:
                        logger.error(f"✗ Numerical stability test failed for {case_name} with {distance_type.value}")
                
                except Exception as e:
                    results[case_name][distance_type.value] = {
                        'status': 'FAILED',
                        'error': str(e)
                    }
                    logger.error(f"✗ Numerical stability test failed for {case_name} with {distance_type.value}: {e}")
        
        self.results['numerical_stability'] = results
        return results
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """Benchmark performance against standard attention."""
        logger.info("Benchmarking performance...")
        
        results = {}
        
        # Create test data
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, self.config.hidden_size, device=self.device)
        
        # Standard multi-head attention for comparison
        standard_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_heads,
            dropout=0.1,
            batch_first=True
        ).to(self.device)
        
        # Test standard attention
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # Multiple runs for average
                _ = standard_attention(x, x, x)[0]
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        standard_time = (time.time() - start_time) / 10
        
        # Test distance-based attention
        for distance_type in [DistanceType.L1, DistanceType.L2, DistanceType.LP]:
            logger.info(f"Benchmarking {distance_type.value} distance attention...")
            
            config = DistanceAttentionConfig(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                distance_type=distance_type.value,
                lambda_param=1.0
            )
            
            distance_attention = create_distance_attention(config).to(self.device)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # Multiple runs for average
                    _ = distance_attention(x, output_attentions=False)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            distance_time = (time.time() - start_time) / 10
            
            # Memory usage comparison
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                _ = distance_attention(x, output_attentions=False)
                distance_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                _ = standard_attention(x, x, x)[0]
                standard_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            else:
                distance_memory = 0
                standard_memory = 0
            
            results[distance_type.value] = {
                'distance_time': distance_time,
                'standard_time': standard_time,
                'speedup': standard_time / distance_time if distance_time > 0 else 0,
                'distance_memory_mb': distance_memory,
                'standard_memory_mb': standard_memory,
                'memory_ratio': distance_memory / standard_memory if standard_memory > 0 else 0
            }
            
            logger.info(f"✓ {distance_type.value} benchmark completed - "
                       f"Time: {distance_time:.4f}s, Speedup: {results[distance_type.value]['speedup']:.2f}x")
        
        self.results['performance_benchmark'] = results
        return results
    
    def test_truthgpt_integration(self) -> Dict[str, Any]:
        """Test integration with TruthGPT model."""
        logger.info("Testing TruthGPT integration...")
        
        results = {}
        
        # Create TruthGPT configuration
        config = TruthGPTAttentionConfig(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            vocab_size=self.config.vocab_size,
            distance_type="l1",
            lambda_param=1.0
        )
        
        # Create model
        model = TruthGPTDistanceAttentionModel(config).to(self.device)
        
        # Test data
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)
        
        try:
            # Test forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True
            )
            
            # Validate outputs
            assert 'logits' in outputs, "Missing logits in outputs"
            assert 'attentions' in outputs, "Missing attentions in outputs"
            assert 'hidden_states' in outputs, "Missing hidden_states in outputs"
            
            assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size), \
                "Incorrect logits shape"
            assert len(outputs['attentions']) == config.num_layers, \
                "Incorrect number of attention layers"
            assert len(outputs['hidden_states']) == config.num_layers + 1, \
                "Incorrect number of hidden states"
            
            # Test gradient flow
            loss = outputs['logits'].sum()
            loss.backward()
            
            # Check that gradients exist
            has_gradients = any(p.grad is not None and torch.any(p.grad != 0) 
                              for p in model.parameters())
            
            results = {
                'status': 'PASSED',
                'logits_shape': outputs['logits'].shape,
                'num_attention_layers': len(outputs['attentions']),
                'num_hidden_states': len(outputs['hidden_states']),
                'has_gradients': has_gradients,
                'model_parameters': sum(p.numel() for p in model.parameters())
            }
            
            logger.info("✓ TruthGPT integration test passed")
            
        except Exception as e:
            results = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ TruthGPT integration test failed: {e}")
        
        self.results['truthgpt_integration'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("Running comprehensive test suite...")
        
        # Run all tests
        self.test_distance_metrics()
        self.test_gradient_flow()
        self.test_numerical_stability()
        self.benchmark_performance()
        self.test_truthgpt_integration()
        
        # Generate summary
        summary = self.generate_test_summary()
        self.results['summary'] = summary
        
        logger.info("All tests completed!")
        return self.results
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_categories': {}
        }
        
        for category, results in self.results.items():
            if category == 'summary':
                continue
                
            category_summary = {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
            
            if isinstance(results, dict):
                for test_name, test_result in results.items():
                    if isinstance(test_result, dict) and 'status' in test_result:
                        category_summary['total'] += 1
                        if test_result['status'] == 'PASSED':
                            category_summary['passed'] += 1
                        else:
                            category_summary['failed'] += 1
                    elif isinstance(test_result, dict):
                        # Handle nested results (like numerical_stability)
                        for sub_test, sub_result in test_result.items():
                            if isinstance(sub_result, dict) and 'status' in sub_result:
                                category_summary['total'] += 1
                                if sub_result['status'] == 'PASSED':
                                    category_summary['passed'] += 1
                                else:
                                    category_summary['failed'] += 1
            
            summary['test_categories'][category] = category_summary
            summary['total_tests'] += category_summary['total']
            summary['passed_tests'] += category_summary['passed']
            summary['failed_tests'] += category_summary['failed']
        
        summary['success_rate'] = summary['passed_tests'] / summary['total_tests'] if summary['total_tests'] > 0 else 0
        
        return summary
    
    def save_results(self, filepath: str):
        """Save test results to file."""
        import json
        
        # Convert torch tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_tensors(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Test results saved to {filepath}")

# Unit test class for unittest framework
class TestDistanceAttention(unittest.TestCase):
    """Unit tests for distance-based attention mechanism."""
    
    def setUp(self):
        self.config = TestConfig(
            batch_size=2,
            seq_len=32,
            hidden_size=256,
            num_heads=4,
            device="cpu"  # Use CPU for unit tests
        )
        self.tester = DistanceAttentionTester(self.config)
    
    def test_l1_distance_attention(self):
        """Test L1 distance attention mechanism."""
        config = DistanceAttentionConfig(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            distance_type="l1"
        )
        
        attention = create_distance_attention(config)
        x = torch.randn(self.config.batch_size, self.config.seq_len, self.config.hidden_size)
        
        output, attn_weights = attention(x, output_attentions=True)
        
        self.assertEqual(output.shape, (self.config.batch_size, self.config.seq_len, self.config.hidden_size))
        self.assertEqual(attn_weights.shape, (self.config.batch_size, self.config.num_heads, self.config.seq_len, self.config.seq_len))
        
        # Check attention weights sum to 1
        attn_sum = attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-6))
    
    def test_l2_distance_attention(self):
        """Test L2 distance attention mechanism."""
        config = DistanceAttentionConfig(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            distance_type="l2"
        )
        
        attention = create_distance_attention(config)
        x = torch.randn(self.config.batch_size, self.config.seq_len, self.config.hidden_size)
        
        output, attn_weights = attention(x, output_attentions=True)
        
        self.assertEqual(output.shape, (self.config.batch_size, self.config.seq_len, self.config.hidden_size))
        self.assertEqual(attn_weights.shape, (self.config.batch_size, self.config.num_heads, self.config.seq_len, self.config.seq_len))
    
    def test_gradient_flow(self):
        """Test gradient flow through attention mechanism."""
        config = DistanceAttentionConfig(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            distance_type="l1"
        )
        
        attention = create_distance_attention(config)
        x = torch.randn(self.config.batch_size, self.config.seq_len, self.config.hidden_size, requires_grad=True)
        
        output, _ = attention(x, output_attentions=False)
        loss = output.sum()
        loss.backward()
        
        self.assertTrue(x.grad is not None)
        self.assertTrue(torch.any(x.grad != 0))

# Main execution
if __name__ == "__main__":
    # Run comprehensive tests
    config = TestConfig()
    tester = DistanceAttentionTester(config)
    
    print("=" * 60)
    print("DISTANCE-BASED ATTENTION COMPREHENSIVE TESTING")
    print("=" * 60)
    
    results = tester.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print(f"\nTEST SUMMARY:")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    
    # Print detailed results
    for category, category_results in summary['test_categories'].items():
        print(f"\n{category.upper()}:")
        print(f"  Total: {category_results['total']}")
        print(f"  Passed: {category_results['passed']}")
        print(f"  Failed: {category_results['failed']}")
    
    # Save results
    tester.save_results("distance_attention_test_results.json")
    
    # Run unit tests
    print(f"\n{'='*60}")
    print("RUNNING UNIT TESTS")
    print(f"{'='*60}")
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")






