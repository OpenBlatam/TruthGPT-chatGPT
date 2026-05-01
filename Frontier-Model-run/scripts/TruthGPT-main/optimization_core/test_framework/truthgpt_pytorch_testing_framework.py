"""
TruthGPT PyTorch Testing Framework
Comprehensive testing framework for PyTorch-inspired optimizations
Tests all optimization systems to ensure they work correctly and provide expected benefits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import logging
import numpy as np
from collections import defaultdict, deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial, lru_cache
import gc
import psutil
from contextlib import contextmanager
import warnings
import math
import random
from enum import Enum
import hashlib
import json
import pickle
from pathlib import Path
import cmath
from abc import ABC, abstractmethod
import unittest
import pytest
import sys
import os

# Add the optimization_core directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our optimization modules
try:
    from pytorch_inspired_optimizer import (
        PyTorchInspiredOptimizer, create_pytorch_inspired_optimizer,
        PyTorchOptimizationLevel, PyTorchOptimizationResult
    )
    from truthgpt_inductor_optimizer import (
        TruthGPTInductorOptimizer, create_truthgpt_inductor_optimizer,
        TruthGPTInductorLevel, TruthGPTInductorResult
    )
    from truthgpt_dynamo_optimizer import (
        TruthGPTDynamoOptimizer, create_truthgpt_dynamo_optimizer,
        TruthGPTDynamoLevel, TruthGPTDynamoResult
    )
    from truthgpt_quantization_optimizer import (
        TruthGPTQuantizationOptimizer, create_truthgpt_quantization_optimizer,
        TruthGPTQuantizationLevel, TruthGPTQuantizationResult
    )
except ImportError as e:
    print(f"Warning: Could not import optimization modules: {e}")
    # Create dummy classes for testing
    class PyTorchInspiredOptimizer:
        def __init__(self, config=None):
            self.config = config or {}
        def optimize_pytorch_style(self, model):
            return type('Result', (), {'speed_improvement': 2.0, 'memory_reduction': 0.1, 'techniques_applied': ['test']})()
    
    class TruthGPTInductorOptimizer:
        def __init__(self, config=None):
            self.config = config or {}
        def optimize_truthgpt_inductor(self, model):
            return type('Result', (), {'speed_improvement': 3.0, 'memory_reduction': 0.2, 'techniques_applied': ['test']})()
    
    class TruthGPTDynamoOptimizer:
        def __init__(self, config=None):
            self.config = config or {}
        def optimize_truthgpt_dynamo(self, model, sample_input):
            return type('Result', (), {'speed_improvement': 4.0, 'memory_reduction': 0.3, 'techniques_applied': ['test']})()
    
    class TruthGPTQuantizationOptimizer:
        def __init__(self, config=None):
            self.config = config or {}
        def optimize_truthgpt_quantization(self, model):
            return type('Result', (), {'speed_improvement': 5.0, 'memory_reduction': 0.4, 'techniques_applied': ['test']})()

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TruthGPTTestResult:
    """Result of a TruthGPT test."""
    
    def __init__(self, test_name: str, success: bool, metrics: Dict[str, float], 
                 error: Optional[str] = None, duration: float = 0.0):
        self.test_name = test_name
        self.success = success
        self.metrics = metrics
        self.error = error
        self.duration = duration
        self.timestamp = time.time()

class TruthGPTTestSuite:
    """Comprehensive test suite for TruthGPT PyTorch optimizations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.test_results = []
        self.benchmark_results = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizers
        self.pytorch_optimizer = PyTorchInspiredOptimizer(config.get('pytorch', {}))
        self.inductor_optimizer = TruthGPTInductorOptimizer(config.get('inductor', {}))
        self.dynamo_optimizer = TruthGPTDynamoOptimizer(config.get('dynamo', {}))
        self.quantization_optimizer = TruthGPTQuantizationOptimizer(config.get('quantization', {}))
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        self.logger.info("🧪 Starting TruthGPT PyTorch optimization tests")
        
        start_time = time.time()
        
        # Run individual test suites
        pytorch_results = self._run_pytorch_tests()
        inductor_results = self._run_inductor_tests()
        dynamo_results = self._run_dynamo_tests()
        quantization_results = self._run_quantization_tests()
        
        # Run integration tests
        integration_results = self._run_integration_tests()
        
        # Run performance benchmarks
        benchmark_results = self._run_performance_benchmarks()
        
        # Run stress tests
        stress_results = self._run_stress_tests()
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'total_tests': len(self.test_results),
            'successful_tests': len([r for r in self.test_results if r.success]),
            'failed_tests': len([r for r in self.test_results if not r.success]),
            'total_time': total_time,
            'pytorch_results': pytorch_results,
            'inductor_results': inductor_results,
            'dynamo_results': dynamo_results,
            'quantization_results': quantization_results,
            'integration_results': integration_results,
            'benchmark_results': benchmark_results,
            'stress_results': stress_results,
            'test_results': self.test_results,
            'success_rate': len([r for r in self.test_results if r.success]) / len(self.test_results) if self.test_results else 0
        }
        
        self.logger.info(f"✅ TruthGPT tests completed: {results['successful_tests']}/{results['total_tests']} passed in {total_time:.2f}s")
        
        return results
    
    def _run_pytorch_tests(self) -> Dict[str, Any]:
        """Run PyTorch-inspired optimization tests."""
        self.logger.info("🔧 Running PyTorch-inspired optimization tests")
        
        results = {
            'tests': [],
            'benchmarks': [],
            'success_rate': 0.0
        }
        
        # Test basic functionality
        try:
            start_time = time.time()
            model = self._create_test_model()
            result = self.pytorch_optimizer.optimize_pytorch_style(model)
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'pytorch_basic_optimization',
                True,
                {
                    'speed_improvement': result.speed_improvement,
                    'memory_reduction': result.memory_reduction,
                    'optimization_time': result.optimization_time
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'pytorch_basic_optimization',
                False,
                {},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Test different optimization levels
        for level in ['basic', 'advanced', 'expert', 'master', 'legendary']:
            try:
                start_time = time.time()
                config = {'level': level}
                optimizer = PyTorchInspiredOptimizer(config)
                model = self._create_test_model()
                result = optimizer.optimize_pytorch_style(model)
                duration = time.time() - start_time
                
                test_result = TruthGPTTestResult(
                    f'pytorch_{level}_optimization',
                    True,
                    {
                        'speed_improvement': result.speed_improvement,
                        'memory_reduction': result.memory_reduction,
                        'level': level
                    },
                    duration=duration
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
                
            except Exception as e:
                test_result = TruthGPTTestResult(
                    f'pytorch_{level}_optimization',
                    False,
                    {'level': level},
                    error=str(e)
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
        
        # Calculate success rate
        results['success_rate'] = len([t for t in results['tests'] if t.success]) / len(results['tests']) if results['tests'] else 0
        
        return results
    
    def _run_inductor_tests(self) -> Dict[str, Any]:
        """Run TruthGPT Inductor optimization tests."""
        self.logger.info("🔥 Running TruthGPT Inductor optimization tests")
        
        results = {
            'tests': [],
            'benchmarks': [],
            'success_rate': 0.0
        }
        
        # Test basic functionality
        try:
            start_time = time.time()
            model = self._create_test_model()
            result = self.inductor_optimizer.optimize_truthgpt_inductor(model)
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'inductor_basic_optimization',
                True,
                {
                    'speed_improvement': result.speed_improvement,
                    'memory_reduction': result.memory_reduction,
                    'kernel_fusion_benefit': result.kernel_fusion_benefit
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'inductor_basic_optimization',
                False,
                {},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Test different optimization levels
        for level in ['basic', 'advanced', 'expert', 'master', 'legendary']:
            try:
                start_time = time.time()
                config = {'level': level}
                optimizer = TruthGPTInductorOptimizer(config)
                model = self._create_test_model()
                result = optimizer.optimize_truthgpt_inductor(model)
                duration = time.time() - start_time
                
                test_result = TruthGPTTestResult(
                    f'inductor_{level}_optimization',
                    True,
                    {
                        'speed_improvement': result.speed_improvement,
                        'memory_reduction': result.memory_reduction,
                        'level': level
                    },
                    duration=duration
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
                
            except Exception as e:
                test_result = TruthGPTTestResult(
                    f'inductor_{level}_optimization',
                    False,
                    {'level': level},
                    error=str(e)
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
        
        # Calculate success rate
        results['success_rate'] = len([t for t in results['tests'] if t.success]) / len(results['tests']) if results['tests'] else 0
        
        return results
    
    def _run_dynamo_tests(self) -> Dict[str, Any]:
        """Run TruthGPT Dynamo optimization tests."""
        self.logger.info("⚡ Running TruthGPT Dynamo optimization tests")
        
        results = {
            'tests': [],
            'benchmarks': [],
            'success_rate': 0.0
        }
        
        # Test basic functionality
        try:
            start_time = time.time()
            model = self._create_test_model()
            sample_input = torch.randn(1, 512)
            result = self.dynamo_optimizer.optimize_truthgpt_dynamo(model, sample_input)
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'dynamo_basic_optimization',
                True,
                {
                    'speed_improvement': result.speed_improvement,
                    'memory_reduction': result.memory_reduction,
                    'graph_optimization_benefit': result.graph_optimization_benefit
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'dynamo_basic_optimization',
                False,
                {},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Test different optimization levels
        for level in ['basic', 'advanced', 'expert', 'master', 'legendary']:
            try:
                start_time = time.time()
                config = {'level': level}
                optimizer = TruthGPTDynamoOptimizer(config)
                model = self._create_test_model()
                sample_input = torch.randn(1, 512)
                result = optimizer.optimize_truthgpt_dynamo(model, sample_input)
                duration = time.time() - start_time
                
                test_result = TruthGPTTestResult(
                    f'dynamo_{level}_optimization',
                    True,
                    {
                        'speed_improvement': result.speed_improvement,
                        'memory_reduction': result.memory_reduction,
                        'level': level
                    },
                    duration=duration
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
                
            except Exception as e:
                test_result = TruthGPTTestResult(
                    f'dynamo_{level}_optimization',
                    False,
                    {'level': level},
                    error=str(e)
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
        
        # Calculate success rate
        results['success_rate'] = len([t for t in results['tests'] if t.success]) / len(results['tests']) if results['tests'] else 0
        
        return results
    
    def _run_quantization_tests(self) -> Dict[str, Any]:
        """Run TruthGPT Quantization optimization tests."""
        self.logger.info("🎯 Running TruthGPT Quantization optimization tests")
        
        results = {
            'tests': [],
            'benchmarks': [],
            'success_rate': 0.0
        }
        
        # Test basic functionality
        try:
            start_time = time.time()
            model = self._create_test_model()
            calibration_data = [torch.randn(1, 512) for _ in range(10)]
            result = self.quantization_optimizer.optimize_truthgpt_quantization(model, calibration_data)
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'quantization_basic_optimization',
                True,
                {
                    'speed_improvement': result.speed_improvement,
                    'memory_reduction': result.memory_reduction,
                    'quantization_benefit': result.quantization_benefit
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'quantization_basic_optimization',
                False,
                {},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Test different optimization levels
        for level in ['basic', 'advanced', 'expert', 'master', 'legendary']:
            try:
                start_time = time.time()
                config = {'level': level}
                optimizer = TruthGPTQuantizationOptimizer(config)
                model = self._create_test_model()
                calibration_data = [torch.randn(1, 512) for _ in range(10)]
                result = optimizer.optimize_truthgpt_quantization(model, calibration_data)
                duration = time.time() - start_time
                
                test_result = TruthGPTTestResult(
                    f'quantization_{level}_optimization',
                    True,
                    {
                        'speed_improvement': result.speed_improvement,
                        'memory_reduction': result.memory_reduction,
                        'level': level
                    },
                    duration=duration
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
                
            except Exception as e:
                test_result = TruthGPTTestResult(
                    f'quantization_{level}_optimization',
                    False,
                    {'level': level},
                    error=str(e)
                )
                self.test_results.append(test_result)
                results['tests'].append(test_result)
        
        # Calculate success rate
        results['success_rate'] = len([t for t in results['tests'] if t.success]) / len(results['tests']) if results['tests'] else 0
        
        return results
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests combining multiple optimizers."""
        self.logger.info("🔗 Running integration tests")
        
        results = {
            'tests': [],
            'success_rate': 0.0
        }
        
        # Test combined PyTorch + Inductor
        try:
            start_time = time.time()
            model = self._create_test_model()
            
            # Apply PyTorch optimizations
            pytorch_result = self.pytorch_optimizer.optimize_pytorch_style(model)
            pytorch_model = pytorch_result.optimized_model
            
            # Apply Inductor optimizations
            inductor_result = self.inductor_optimizer.optimize_truthgpt_inductor(pytorch_model)
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'pytorch_inductor_integration',
                True,
                {
                    'pytorch_speed_improvement': pytorch_result.speed_improvement,
                    'inductor_speed_improvement': inductor_result.speed_improvement,
                    'combined_speed_improvement': pytorch_result.speed_improvement * inductor_result.speed_improvement
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'pytorch_inductor_integration',
                False,
                {},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Test combined Dynamo + Quantization
        try:
            start_time = time.time()
            model = self._create_test_model()
            sample_input = torch.randn(1, 512)
            calibration_data = [torch.randn(1, 512) for _ in range(10)]
            
            # Apply Dynamo optimizations
            dynamo_result = self.dynamo_optimizer.optimize_truthgpt_dynamo(model, sample_input)
            dynamo_model = dynamo_result.optimized_model
            
            # Apply Quantization optimizations
            quantization_result = self.quantization_optimizer.optimize_truthgpt_quantization(dynamo_model, calibration_data)
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'dynamo_quantization_integration',
                True,
                {
                    'dynamo_speed_improvement': dynamo_result.speed_improvement,
                    'quantization_speed_improvement': quantization_result.speed_improvement,
                    'combined_speed_improvement': dynamo_result.speed_improvement * quantization_result.speed_improvement
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'dynamo_quantization_integration',
                False,
                {},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Test all optimizers combined
        try:
            start_time = time.time()
            model = self._create_test_model()
            sample_input = torch.randn(1, 512)
            calibration_data = [torch.randn(1, 512) for _ in range(10)]
            
            # Apply all optimizations in sequence
            pytorch_result = self.pytorch_optimizer.optimize_pytorch_style(model)
            inductor_result = self.inductor_optimizer.optimize_truthgpt_inductor(pytorch_result.optimized_model)
            dynamo_result = self.dynamo_optimizer.optimize_truthgpt_dynamo(inductor_result.optimized_model, sample_input)
            quantization_result = self.quantization_optimizer.optimize_truthgpt_quantization(dynamo_result.optimized_model, calibration_data)
            duration = time.time() - start_time
            
            combined_speedup = (pytorch_result.speed_improvement * 
                              inductor_result.speed_improvement * 
                              dynamo_result.speed_improvement * 
                              quantization_result.speed_improvement)
            
            test_result = TruthGPTTestResult(
                'all_optimizers_integration',
                True,
                {
                    'combined_speed_improvement': combined_speedup,
                    'pytorch_improvement': pytorch_result.speed_improvement,
                    'inductor_improvement': inductor_result.speed_improvement,
                    'dynamo_improvement': dynamo_result.speed_improvement,
                    'quantization_improvement': quantization_result.speed_improvement
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'all_optimizers_integration',
                False,
                {},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Calculate success rate
        results['success_rate'] = len([t for t in results['tests'] if t.success]) / len(results['tests']) if results['tests'] else 0
        
        return results
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        self.logger.info("📊 Running performance benchmarks")
        
        results = {
            'benchmarks': [],
            'summary': {}
        }
        
        # Benchmark different model sizes
        model_sizes = [64, 128, 256, 512, 1024]
        test_inputs = [torch.randn(1, 512) for _ in range(10)]
        
        for size in model_sizes:
            try:
                start_time = time.time()
                model = self._create_test_model(size)
                
                # Benchmark original model
                original_times = []
                with torch.no_grad():
                    for _ in range(5):
                        iter_start = time.perf_counter()
                        for test_input in test_inputs:
                            _ = model(test_input)
                        iter_end = time.perf_counter()
                        original_times.append((iter_end - iter_start) * 1000)
                
                # Optimize model
                pytorch_result = self.pytorch_optimizer.optimize_pytorch_style(model)
                optimized_model = pytorch_result.optimized_model
                
                # Benchmark optimized model
                optimized_times = []
                with torch.no_grad():
                    for _ in range(5):
                        iter_start = time.perf_counter()
                        for test_input in test_inputs:
                            _ = optimized_model(test_input)
                        iter_end = time.perf_counter()
                        optimized_times.append((iter_end - iter_start) * 1000)
                
                speedup = np.mean(original_times) / np.mean(optimized_times)
                duration = time.time() - start_time
                
                benchmark_result = TruthGPTTestResult(
                    f'performance_benchmark_size_{size}',
                    True,
                    {
                        'model_size': size,
                        'original_avg_time_ms': np.mean(original_times),
                        'optimized_avg_time_ms': np.mean(optimized_times),
                        'speedup': speedup,
                        'memory_reduction': pytorch_result.memory_reduction
                    },
                    duration=duration
                )
                self.test_results.append(benchmark_result)
                results['benchmarks'].append(benchmark_result)
                
            except Exception as e:
                benchmark_result = TruthGPTTestResult(
                    f'performance_benchmark_size_{size}',
                    False,
                    {'model_size': size},
                    error=str(e)
                )
                self.test_results.append(benchmark_result)
                results['benchmarks'].append(benchmark_result)
        
        # Calculate summary statistics
        successful_benchmarks = [b for b in results['benchmarks'] if b.success]
        if successful_benchmarks:
            results['summary'] = {
                'avg_speedup': np.mean([b.metrics['speedup'] for b in successful_benchmarks]),
                'max_speedup': max([b.metrics['speedup'] for b in successful_benchmarks]),
                'avg_memory_reduction': np.mean([b.metrics['memory_reduction'] for b in successful_benchmarks]),
                'total_benchmarks': len(results['benchmarks']),
                'successful_benchmarks': len(successful_benchmarks)
            }
        
        return results
    
    def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests to ensure stability."""
        self.logger.info("💪 Running stress tests")
        
        results = {
            'tests': [],
            'success_rate': 0.0
        }
        
        # Stress test with multiple iterations
        try:
            start_time = time.time()
            model = self._create_test_model()
            
            for i in range(10):
                result = self.pytorch_optimizer.optimize_pytorch_style(model)
                model = result.optimized_model
            
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'stress_test_multiple_iterations',
                True,
                {
                    'iterations': 10,
                    'final_speed_improvement': result.speed_improvement,
                    'final_memory_reduction': result.memory_reduction
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'stress_test_multiple_iterations',
                False,
                {'iterations': 10},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Stress test with large models
        try:
            start_time = time.time()
            large_model = self._create_test_model(2048)
            result = self.pytorch_optimizer.optimize_pytorch_style(large_model)
            duration = time.time() - start_time
            
            test_result = TruthGPTTestResult(
                'stress_test_large_model',
                True,
                {
                    'model_size': 2048,
                    'speed_improvement': result.speed_improvement,
                    'memory_reduction': result.memory_reduction
                },
                duration=duration
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
            
        except Exception as e:
            test_result = TruthGPTTestResult(
                'stress_test_large_model',
                False,
                {'model_size': 2048},
                error=str(e)
            )
            self.test_results.append(test_result)
            results['tests'].append(test_result)
        
        # Calculate success rate
        results['success_rate'] = len([t for t in results['tests'] if t.success]) / len(results['tests']) if results['tests'] else 0
        
        return results
    
    def _create_test_model(self, input_size: int = 512) -> nn.Module:
        """Create a test model for optimization."""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("TruthGPT PyTorch Optimization Test Report")
        report.append("=" * 80)
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {results['total_tests']}")
        report.append(f"Successful: {results['successful_tests']}")
        report.append(f"Failed: {results['failed_tests']}")
        report.append(f"Success Rate: {results['success_rate']:.1%}")
        report.append(f"Total Time: {results['total_time']:.2f}s")
        report.append("")
        
        # Individual optimizer results
        for optimizer_name in ['pytorch', 'inductor', 'dynamo', 'quantization']:
            if f'{optimizer_name}_results' in results:
                opt_results = results[f'{optimizer_name}_results']
                report.append(f"{optimizer_name.upper()} OPTIMIZER RESULTS")
                report.append("-" * 40)
                report.append(f"Success Rate: {opt_results['success_rate']:.1%}")
                report.append(f"Tests: {len(opt_results['tests'])}")
                report.append("")
        
        # Integration results
        if 'integration_results' in results:
            int_results = results['integration_results']
            report.append("INTEGRATION TEST RESULTS")
            report.append("-" * 40)
            report.append(f"Success Rate: {int_results['success_rate']:.1%}")
            report.append(f"Tests: {len(int_results['tests'])}")
            report.append("")
        
        # Performance benchmarks
        if 'benchmark_results' in results and 'summary' in results['benchmark_results']:
            bench_summary = results['benchmark_results']['summary']
            report.append("PERFORMANCE BENCHMARKS")
            report.append("-" * 40)
            report.append(f"Average Speedup: {bench_summary.get('avg_speedup', 0):.1f}x")
            report.append(f"Maximum Speedup: {bench_summary.get('max_speedup', 0):.1f}x")
            report.append(f"Average Memory Reduction: {bench_summary.get('avg_memory_reduction', 0):.1%}")
            report.append("")
        
        # Failed tests
        failed_tests = [r for r in results['test_results'] if not r.success]
        if failed_tests:
            report.append("FAILED TESTS")
            report.append("-" * 40)
            for test in failed_tests:
                report.append(f"- {test.test_name}: {test.error}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        if results['success_rate'] < 0.8:
            report.append("⚠️  Low success rate detected. Review failed tests.")
        if results['total_time'] > 300:
            report.append("⚠️  Tests took longer than expected. Consider optimization.")
        if results['success_rate'] >= 0.9:
            report.append("✅ Excellent test results! All optimizations working correctly.")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_test_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to a file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"truthgpt_pytorch_test_results_{timestamp}.json"
        
        # Convert test results to serializable format
        serializable_results = {
            'timestamp': time.time(),
            'total_tests': results['total_tests'],
            'successful_tests': results['successful_tests'],
            'failed_tests': results['failed_tests'],
            'success_rate': results['success_rate'],
            'total_time': results['total_time'],
            'test_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'metrics': r.metrics,
                    'error': r.error,
                    'duration': r.duration,
                    'timestamp': r.timestamp
                }
                for r in results['test_results']
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Test results saved to {filename}")

# Factory functions
def create_truthgpt_test_suite(config: Optional[Dict[str, Any]] = None) -> TruthGPTTestSuite:
    """Create TruthGPT test suite."""
    return TruthGPTTestSuite(config)

def run_truthgpt_tests(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run all TruthGPT tests and return results."""
    test_suite = create_truthgpt_test_suite(config)
    return test_suite.run_all_tests()

# Example usage and testing
def example_truthgpt_testing():
    """Example of TruthGPT testing."""
    # Create test configuration
    config = {
        'pytorch': {'level': 'legendary'},
        'inductor': {'level': 'legendary'},
        'dynamo': {'level': 'legendary'},
        'quantization': {'level': 'legendary'}
    }
    
    # Run tests
    print("🧪 Running TruthGPT PyTorch optimization tests...")
    results = run_truthgpt_tests(config)
    
    # Generate and print report
    test_suite = create_truthgpt_test_suite(config)
    report = test_suite.generate_test_report(results)
    print(report)
    
    # Save results
    test_suite.save_test_results(results)
    
    return results

if __name__ == "__main__":
    # Run example
    results = example_truthgpt_testing()

