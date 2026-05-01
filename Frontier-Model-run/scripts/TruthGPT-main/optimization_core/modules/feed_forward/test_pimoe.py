"""
Comprehensive test suite for PiMoE token-level routing implementation
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import unittest
from unittest.mock import Mock, patch

from .pimoe_router import (
    PiMoESystem,
    TokenLevelRouter,
    PiMoEExpert,
    ExpertType,
    RoutingDecision,
    create_pimoe_system
)
from .enhanced_pimoe_integration import (
    EnhancedPiMoEIntegration,
    AdaptivePiMoE,
    create_enhanced_pimoe_integration
)

class TestPiMoERouter(unittest.TestCase):
    """Test cases for PiMoE router functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 32
        self.expert_types = [
            ExpertType.REASONING,
            ExpertType.COMPUTATION,
            ExpertType.MATHEMATICAL,
            ExpertType.LOGICAL
        ]
        
        # Create test data
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
    
    def test_token_level_router_initialization(self):
        """Test TokenLevelRouter initialization."""
        router = TokenLevelRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        self.assertEqual(router.hidden_size, self.hidden_size)
        self.assertEqual(router.num_experts, self.num_experts)
        self.assertEqual(router.expert_types, self.expert_types)
        self.assertIsNotNone(router.router_network)
        self.assertIsNotNone(router.expert_type_classifier)
    
    def test_token_level_router_forward(self):
        """Test TokenLevelRouter forward pass."""
        router = TokenLevelRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        # Test basic forward pass
        output = router(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test forward pass with routing info
        output, routing_info = router(self.test_input, return_routing_info=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('routing_decisions', routing_info)
        self.assertIn('expert_probs', routing_info)
        self.assertIn('load_balance_loss', routing_info)
    
    def test_routing_decisions_structure(self):
        """Test that routing decisions have correct structure."""
        router = TokenLevelRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        output, routing_info = router(self.test_input, return_routing_info=True)
        routing_decisions = routing_info['routing_decisions']
        
        # Check that we have the right number of decisions
        expected_decisions = self.batch_size * self.seq_len
        self.assertEqual(len(routing_decisions), expected_decisions)
        
        # Check structure of first decision
        first_decision = routing_decisions[0]
        self.assertIsInstance(first_decision, RoutingDecision)
        self.assertIsInstance(first_decision.token_id, int)
        self.assertIsInstance(first_decision.expert_id, int)
        self.assertIsInstance(first_decision.expert_type, ExpertType)
        self.assertIsInstance(first_decision.confidence, float)
        self.assertIsInstance(first_decision.routing_score, float)
        self.assertIsInstance(first_decision.load_balance_weight, float)
    
    def test_expert_usage_tracking(self):
        """Test expert usage tracking functionality."""
        router = TokenLevelRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        # Run multiple forward passes
        for _ in range(5):
            router(self.test_input)
        
        # Check usage statistics
        stats = router.get_expert_usage_stats()
        self.assertIn('expert_usage_counts', stats)
        self.assertIn('total_usage', stats)
        self.assertIn('load_balance_ratio', stats)
        
        # Check that usage counts are updated
        total_usage = stats['total_usage']
        self.assertGreater(total_usage, 0)
    
    def test_load_balance_calculation(self):
        """Test load balance calculation."""
        router = TokenLevelRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        # Run forward pass
        router(self.test_input)
        
        # Check load balance ratio
        stats = router.get_expert_usage_stats()
        load_balance_ratio = stats['load_balance_ratio']
        self.assertGreaterEqual(load_balance_ratio, 0.0)
        self.assertLessEqual(load_balance_ratio, 1.0)

class TestPiMoEExpert(unittest.TestCase):
    """Test cases for PiMoE expert functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.expert_types = [
            ExpertType.REASONING,
            ExpertType.COMPUTATION,
            ExpertType.MATHEMATICAL,
            ExpertType.LOGICAL
        ]
        self.test_input = torch.randn(10, self.hidden_size)
    
    def test_expert_initialization(self):
        """Test expert initialization for different types."""
        for expert_type in self.expert_types:
            expert = PiMoEExpert(
                hidden_size=self.hidden_size,
                expert_type=expert_type
            )
            
            self.assertEqual(expert.hidden_size, self.hidden_size)
            self.assertEqual(expert.expert_type, expert_type)
            self.assertIsNotNone(expert.layers)
    
    def test_expert_forward_pass(self):
        """Test expert forward pass."""
        expert = PiMoEExpert(
            hidden_size=self.hidden_size,
            expert_type=ExpertType.REASONING
        )
        
        output = expert(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
    
    def test_different_expert_architectures(self):
        """Test that different expert types have different architectures."""
        experts = {}
        
        for expert_type in self.expert_types:
            expert = PiMoEExpert(
                hidden_size=self.hidden_size,
                expert_type=expert_type
            )
            experts[expert_type] = expert
        
        # Test that outputs are different for different expert types
        outputs = {}
        for expert_type, expert in experts.items():
            outputs[expert_type] = expert(self.test_input)
        
        # Check that outputs are different (not identical)
        for i, (type1, output1) in enumerate(outputs.items()):
            for j, (type2, output2) in enumerate(outputs.items()):
                if i != j:
                    # Outputs should not be identical
                    self.assertFalse(torch.allclose(output1, output2, atol=1e-6))

class TestPiMoESystem(unittest.TestCase):
    """Test cases for complete PiMoE system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 32
        self.expert_types = [
            ExpertType.REASONING,
            ExpertType.COMPUTATION,
            ExpertType.MATHEMATICAL,
            ExpertType.LOGICAL
        ]
        
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
    
    def test_pimoe_system_initialization(self):
        """Test PiMoE system initialization."""
        system = PiMoESystem(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        self.assertEqual(system.hidden_size, self.hidden_size)
        self.assertEqual(system.num_experts, self.num_experts)
        self.assertEqual(len(system.experts), self.num_experts)
        self.assertIsNotNone(system.router)
    
    def test_pimoe_system_forward(self):
        """Test PiMoE system forward pass."""
        system = PiMoESystem(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        # Test basic forward pass
        output = system(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test forward pass with routing info
        output, routing_info = system(self.test_input, return_routing_info=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('routing_decisions', routing_info)
        self.assertIn('expert_probs', routing_info)
    
    def test_create_pimoe_system_factory(self):
        """Test PiMoE system factory function."""
        system = create_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        self.assertIsInstance(system, PiMoESystem)
        self.assertEqual(system.hidden_size, self.hidden_size)
        self.assertEqual(system.num_experts, self.num_experts)
    
    def test_system_statistics(self):
        """Test system statistics functionality."""
        system = PiMoESystem(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        # Run forward pass to generate statistics
        system(self.test_input)
        
        # Get system statistics
        stats = system.get_system_stats()
        self.assertIn('router_stats', stats)
        self.assertIn('num_experts', stats)
        self.assertIn('expert_types', stats)
        self.assertIn('system_efficiency', stats)

class TestEnhancedPiMoEIntegration(unittest.TestCase):
    """Test cases for enhanced PiMoE integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 32
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_enhanced_integration_initialization(self):
        """Test enhanced integration initialization."""
        integration = EnhancedPiMoEIntegration(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            optimization_level="advanced"
        )
        
        self.assertEqual(integration.hidden_size, self.hidden_size)
        self.assertEqual(integration.num_experts, self.num_experts)
        self.assertIsNotNone(integration.pimoe_system)
    
    def test_enhanced_integration_forward(self):
        """Test enhanced integration forward pass."""
        integration = EnhancedPiMoEIntegration(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        # Test basic forward pass
        output = integration(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test forward pass with metrics
        output, metrics = integration(self.test_input, return_metrics=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('latency_ms', metrics)
        self.assertIn('throughput_tokens_per_sec', metrics)
        self.assertIn('memory_usage_mb', metrics)
    
    def test_optimization_report(self):
        """Test optimization report generation."""
        integration = EnhancedPiMoEIntegration(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        # Run forward pass to generate data
        integration(self.test_input)
        
        # Get optimization report
        report = integration.get_optimization_report()
        self.assertIn('system_stats', report)
        self.assertIn('optimization_config', report)
        self.assertIn('performance_metrics', report)
        self.assertIn('recommendations', report)

class TestAdaptivePiMoE(unittest.TestCase):
    """Test cases for adaptive PiMoE system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 32
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_adaptive_pimoe_initialization(self):
        """Test adaptive PiMoE initialization."""
        adaptive_system = AdaptivePiMoE(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        self.assertEqual(adaptive_system.hidden_size, self.hidden_size)
        self.assertEqual(adaptive_system.num_experts, self.num_experts)
        self.assertIsNotNone(adaptive_system.pimoe_system)
    
    def test_adaptive_forward_pass(self):
        """Test adaptive forward pass."""
        adaptive_system = AdaptivePiMoE(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        # Test basic forward pass
        output = adaptive_system(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test forward pass with adaptation info
        output, info = adaptive_system(self.test_input, return_adaptation_info=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('routing_info', info)
        self.assertIn('metrics', info)
        self.assertIn('adaptation_info', info)
    
    def test_adaptation_mechanism(self):
        """Test adaptation mechanism."""
        adaptive_system = AdaptivePiMoE(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            performance_threshold=0.9  # High threshold to trigger adaptation
        )
        
        # Run multiple iterations to test adaptation
        for _ in range(5):
            output, info = adaptive_system(self.test_input, return_adaptation_info=True)
            adaptation_info = info['adaptation_info']
            
            # Check adaptation info structure
            self.assertIn('adaptation_applied', adaptation_info)
            self.assertIn('performance_score', adaptation_info)
            self.assertIn('changes', adaptation_info)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 512
        self.num_experts = 8
        self.batch_size = 4
        self.seq_len = 128
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_latency_benchmark(self):
        """Test latency performance."""
        system = create_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        # Warm up
        for _ in range(5):
            system(self.test_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            system(self.test_input)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / 10 * 1000  # Convert to ms
        self.assertLess(avg_latency, 1000)  # Should be less than 1 second
    
    def test_memory_usage(self):
        """Test memory usage."""
        system = create_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        # Get initial memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Run forward pass
        output = system(self.test_input)
        
        # Get final memory usage
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory usage should be reasonable
        memory_usage = final_memory - initial_memory
        if torch.cuda.is_available():
            self.assertLess(memory_usage, 100 * 1024 * 1024)  # Less than 100MB
    
    def test_throughput_benchmark(self):
        """Test throughput performance."""
        system = create_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        # Warm up
        for _ in range(5):
            system(self.test_input)
        
        # Benchmark throughput
        start_time = time.time()
        num_iterations = 20
        
        for _ in range(num_iterations):
            system(self.test_input)
        
        end_time = time.time()
        
        total_tokens = self.batch_size * self.seq_len * num_iterations
        throughput = total_tokens / (end_time - start_time)
        
        self.assertGreater(throughput, 1000)  # Should process at least 1000 tokens/sec

class TestIntegrationWithTruthGPT(unittest.TestCase):
    """Test integration with TruthGPT optimization core."""
    
    def test_import_compatibility(self):
        """Test that all components can be imported correctly."""
        try:
            from .pimoe_router import (
                PiMoESystem,
                TokenLevelRouter,
                ExpertType,
                create_pimoe_system
            )
            from .enhanced_pimoe_integration import (
                EnhancedPiMoEIntegration,
                AdaptivePiMoE,
                create_enhanced_pimoe_integration
            )
            from .pimoe_demo import PiMoEDemo, DemoConfig
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_factory_function_compatibility(self):
        """Test factory function compatibility."""
        # Test basic PiMoE system creation
        basic_system = create_pimoe_system(
            hidden_size=256,
            num_experts=4
        )
        self.assertIsNotNone(basic_system)
        
        # Test enhanced integration creation
        enhanced_system = create_enhanced_pimoe_integration(
            hidden_size=256,
            num_experts=4,
            optimization_level="advanced"
        )
        self.assertIsNotNone(enhanced_system)
        
        # Test adaptive system creation
        adaptive_system = create_enhanced_pimoe_integration(
            hidden_size=256,
            num_experts=4,
            enable_adaptation=True
        )
        self.assertIsNotNone(adaptive_system)

def run_all_tests():
    """Run all test suites."""
    test_suites = [
        TestPiMoERouter,
        TestPiMoEExpert,
        TestPiMoESystem,
        TestEnhancedPiMoEIntegration,
        TestAdaptivePiMoE,
        TestPerformanceBenchmarks,
        TestIntegrationWithTruthGPT
    ]
    
    all_tests = unittest.TestSuite()
    
    for test_suite in test_suites:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        all_tests.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    return result

if __name__ == "__main__":
    # Run all tests
    result = run_all_tests()
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        for failure in result.failures:
            print(f"FAIL: {failure[0]}")
            print(f"  {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"  {error[1]}")





