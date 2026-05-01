"""
End-to-end integration tests for TruthGPT optimization core
Tests complete workflows and system integration
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.fixtures.test_data import TestDataFactory
from tests.fixtures.mock_components import MockModel, MockOptimizer, MockAttention, MockMLP, MockKVCache, MockDataset
from tests.fixtures.test_utils import TestUtils, PerformanceProfiler, TestAssertions

class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = TestDataFactory()
        self.profiler = PerformanceProfiler()
        self.config = TestUtils.create_test_config()
        
    def test_complete_training_workflow(self):
        """Test complete training workflow"""
        # Create components
        model = MockModel(input_size=512, hidden_size=1024, output_size=512)
        optimizer = MockOptimizer(learning_rate=0.001)
        dataset = MockDataset(size=100, input_size=512, output_size=512)
        
        # Training loop
        num_epochs = 5
        batch_size = 8
        
        for epoch in range(num_epochs):
            # Get batch
            batch = dataset.get_batch(batch_size)
            
            # Forward pass
            output = model(batch['input'])
            
            # Compute loss
            target = batch['target']
            loss = nn.MSELoss()(output, target)
            
            # Optimizer step
            result = optimizer.step(loss)
            
            # Verify step
            self.assertTrue(result['optimized'])
            
        # Verify final state
        self.assertEqual(optimizer.step_count, num_epochs)
        self.assertEqual(model.forward_count, num_epochs)
        
    def test_attention_mlp_integration_workflow(self):
        """Test attention-MLP integration workflow"""
        # Create components
        attention = MockAttention(d_model=512, n_heads=8)
        mlp = MockMLP(input_size=512, hidden_size=2048, output_size=512)
        cache = MockKVCache(max_size=1000)
        
        # Create test data
        attention_data = self.test_data.create_attention_data()
        
        # Attention forward pass
        attn_output, weights = attention(attention_data['query'], 
                                        attention_data['key'], 
                                        attention_data['value'])
        
        # Cache attention output
        cache.put("attention_output", attn_output)
        
        # MLP forward pass
        mlp_output = mlp(attn_output)
        
        # Verify integration
        self.assertEqual(mlp_output.shape, attn_output.shape)
        
        # Verify cache
        cached_output = cache.get("attention_output")
        self.assertIsNotNone(cached_output)
        self.assertTrue(torch.equal(cached_output, attn_output))
        
    def test_optimization_pipeline_workflow(self):
        """Test complete optimization pipeline"""
        # Create components
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        optimizer = MockOptimizer(learning_rate=0.001)
        dataset = MockDataset(size=50, input_size=256, output_size=256)
        
        # Optimization pipeline
        self.profiler.start_profile("optimization_pipeline")
        
        # Training phase
        for epoch in range(3):
            batch = dataset.get_batch(4)
            output = model(batch['input'])
            target = batch['target']
            loss = nn.MSELoss()(output, target)
            optimizer.step(loss)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            eval_batch = dataset.get_batch(4)
            eval_output = model(eval_batch['input'])
            eval_loss = nn.MSELoss()(eval_output, eval_batch['target'])
        
        metrics = self.profiler.end_profile()
        
        # Verify pipeline
        self.assertLess(metrics['execution_time'], 10.0)
        self.assertIsInstance(eval_loss.item(), float)
        
    def test_multi_component_integration(self):
        """Test multi-component integration"""
        # Create all components
        attention = MockAttention(d_model=512, n_heads=8)
        mlp = MockMLP(input_size=512, hidden_size=2048, output_size=512)
        model = MockModel(input_size=512, hidden_size=1024, output_size=512)
        optimizer = MockOptimizer(learning_rate=0.001)
        cache = MockKVCache(max_size=1000)
        dataset = MockDataset(size=20, input_size=512, output_size=512)
        
        # Integration workflow
        for step in range(3):
            # Get data
            batch = dataset.get_batch(2)
            attention_data = self.test_data.create_attention_data()
            
            # Attention processing
            attn_output, weights = attention(attention_data['query'], 
                                            attention_data['key'], 
                                            attention_data['value'])
            
            # Cache attention
            cache.put(f"attention_{step}", attn_output)
            
            # MLP processing
            mlp_output = mlp(attn_output)
            
            # Model processing
            model_output = model(batch['input'])
            
            # Optimization
            target = batch['target']
            loss = nn.MSELoss()(model_output, target)
            optimizer.step(loss)
            
        # Verify all components worked
        self.assertGreater(attention.get_attention_stats()['attention_count'], 0)
        self.assertGreater(mlp.get_mlp_stats()['forward_count'], 0)
        self.assertGreater(model.get_model_stats()['forward_count'], 0)
        self.assertGreater(optimizer.get_optimization_stats()['total_steps'], 0)
        self.assertGreater(cache.get_stats()['hit_count'], 0)
        
    def test_performance_benchmark_workflow(self):
        """Test performance benchmark workflow"""
        # Create components
        model = MockModel(input_size=512, hidden_size=1024, output_size=512)
        optimizer = MockOptimizer(learning_rate=0.001)
        
        # Performance benchmark
        benchmark_data = self.test_data.create_benchmark_data()
        
        results = []
        for data in benchmark_data:
            # Create input
            input_tensor = torch.randn(2, data['seq_len'], data['d_model'])
            
            # Profile forward pass
            self.profiler.start_profile(f"benchmark_{data['seq_len']}")
            output = model(input_tensor)
            metrics = self.profiler.end_profile()
            
            results.append({
                'seq_len': data['seq_len'],
                'execution_time': metrics['execution_time'],
                'memory_used': metrics['memory_used']
            })
        
        # Verify benchmark results
        self.assertEqual(len(results), len(benchmark_data))
        for result in results:
            self.assertGreater(result['execution_time'], 0)
            self.assertGreater(result['memory_used'], 0)
            
    def test_error_recovery_workflow(self):
        """Test error recovery workflow"""
        model = MockModel(input_size=256, hidden_size=512, output_size=256)
        optimizer = MockOptimizer(learning_rate=0.001)
        
        # Test error cases
        error_cases = self.test_data.create_error_cases()
        
        for case in error_cases:
            with self.subTest(case=case['name']):
                try:
                    # Attempt to process problematic data
                    if case['name'] == 'invalid_tensor_shape':
                        # This should be handled gracefully
                        pass
                    elif case['name'] == 'negative_dimensions':
                        # This should raise an error
                        with self.assertRaises(RuntimeError):
                            torch.randn(-1, 128)
                    elif case['name'] == 'mismatched_batch_sizes':
                        # This should be handled gracefully
                        pass
                        
                except Exception as e:
                    # Verify error is expected type
                    self.assertIsInstance(e, case['expected_error'])
                    
    def test_memory_management_workflow(self):
        """Test memory management workflow"""
        model = MockModel(input_size=512, hidden_size=1024, output_size=512)
        cache = MockKVCache(max_size=100)
        
        # Test memory management
        for i in range(150):  # Exceed cache size
            data = torch.randn(1, 8, 64)
            cache.put(f"key_{i}", data)
            
        # Verify cache management
        stats = cache.get_stats()
        self.assertLessEqual(stats['cache_size'], cache.max_size)
        
        # Test cache hits/misses
        self.assertGreater(stats['hit_count'] + stats['miss_count'], 0)
        
    def test_scalability_workflow(self):
        """Test scalability workflow"""
        # Test different model sizes
        model_sizes = [(128, 256, 128), (256, 512, 256), (512, 1024, 512)]
        
        for input_size, hidden_size, output_size in model_sizes:
            with self.subTest(size=(input_size, hidden_size, output_size)):
                model = MockModel(input_size, hidden_size, output_size)
                optimizer = MockOptimizer(learning_rate=0.001)
                
                # Test with different batch sizes
                batch_sizes = [1, 2, 4, 8]
                
                for batch_size in batch_sizes:
                    input_data = torch.randn(batch_size, 64, input_size)
                    
                    # Profile forward pass
                    self.profiler.start_profile(f"scale_{input_size}_{batch_size}")
                    output = model(input_data)
                    metrics = self.profiler.end_profile()
                    
                    # Verify scalability
                    self.assertEqual(output.shape, (batch_size, 64, output_size))
                    self.assertLess(metrics['execution_time'], 5.0)  # Should be reasonable

class TestSystemIntegration(unittest.TestCase):
    """Test system-level integration"""
    
    def setUp(self):
        """Set up system test fixtures"""
        self.test_data = TestDataFactory()
        
    def test_system_initialization(self):
        """Test system initialization"""
        # Test all components can be initialized
        components = {
            'model': MockModel(),
            'optimizer': MockOptimizer(),
            'attention': MockAttention(),
            'mlp': MockMLP(),
            'cache': MockKVCache(),
            'dataset': MockDataset()
        }
        
        # Verify all components initialized successfully
        for name, component in components.items():
            self.assertIsNotNone(component, f"{name} failed to initialize")
            
    def test_system_communication(self):
        """Test system component communication"""
        # Create system
        attention = MockAttention()
        mlp = MockMLP()
        cache = MockKVCache()
        
        # Test communication flow
        data = self.test_data.create_attention_data()
        
        # Attention -> Cache -> MLP flow
        attn_output, weights = attention(data['query'], data['key'], data['value'])
        cache.put("intermediate", attn_output)
        cached_output = cache.get("intermediate")
        mlp_output = mlp(cached_output)
        
        # Verify communication
        self.assertIsNotNone(attn_output)
        self.assertIsNotNone(cached_output)
        self.assertIsNotNone(mlp_output)
        
    def test_system_performance(self):
        """Test system performance"""
        # Create full system
        model = MockModel()
        optimizer = MockOptimizer()
        dataset = MockDataset(size=10)
        
        # System performance test
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time and end_time:
            start_time.record()
        
        # Run system
        for i in range(3):
            batch = dataset.get_batch(2)
            output = model(batch['input'])
            target = batch['target']
            loss = nn.MSELoss()(output, target)
            optimizer.step(loss)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            execution_time = start_time.elapsed_time(end_time)
            self.assertLess(execution_time, 1000)  # Should be fast

if __name__ == '__main__':
    unittest.main()





