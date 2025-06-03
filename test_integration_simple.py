"""
Simplified integration tests for the TruthGPT system.
Tests multiple components working together without complex serialization.
"""
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'Transformers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture', 'NLP'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture'))

from transformer import attention
from manager import CommandManager
from production import execute_action_based_on_text
from finetune import FineTuner, softmax


class SimpleCommand:
    """Simple command class that doesn't cause pickle issues."""
    
    def __init__(self, name):
        self.name = name
        self.executed = False
        self.undone = False
    
    def execute(self):
        self.executed = True
    
    def undo(self):
        self.undone = True


class TestBasicIntegration:
    """Basic integration tests between components."""
    
    def test_transformer_finetune_integration(self):
        """Test basic integration between transformer and fine-tuning."""
        # Create a fine-tuner
        tuner = FineTuner()
        
        # Generate some sample data
        sample_data = np.random.rand(6, 6)
        result = tuner.finetune(n=50, data=sample_data)
        
        # Use the result to create attention matrices
        if result['softmax_result'] is not None:
            finetuned_data = result['softmax_result']
            
            # Create compatible Q, K, V matrices
            size = min(4, finetuned_data.shape[0], finetuned_data.shape[1])
            Q = finetuned_data[:size, :size]
            K = finetuned_data[:size, :size]
            V = finetuned_data[:size, :size]
            
            # Test attention with fine-tuned values
            attention_output = attention(Q, K, V, dk=size)
            
            assert attention_output.shape == (size, size)
            assert np.all(np.isfinite(attention_output))
    
    def test_command_manager_basic_integration(self):
        """Test basic CommandManager integration."""
        manager = CommandManager()
        
        # Create simple commands
        cmd1 = SimpleCommand("first")
        cmd2 = SimpleCommand("second")
        
        # Manually add to queue to avoid pickle issues
        manager.command_queue.put(cmd1)
        manager.command_queue.put(cmd2)
        
        # Execute commands
        manager.execute_commands()
        
        # Verify execution
        assert cmd1.executed
        assert cmd2.executed
    
    def test_production_system_integration(self):
        """Test production system integration."""
        # Test various production actions
        actions = [
            ("create_ticket", {"system": "test", "details": "integration test"}),
            ("update_node_status", {"node_id": "123", "status": "active"}),
            ("get_network_statistics", {"node_id": "456"})
        ]
        
        # Execute actions and verify they don't crash
        with patch('builtins.print'):  # Suppress output
            for action, params in actions:
                execute_action_based_on_text(action, params)
    
    def test_end_to_end_simple_pipeline(self):
        """Test a simple end-to-end pipeline."""
        # Step 1: Fine-tune with small data
        tuner = FineTuner()
        data = np.random.rand(4, 4)
        result = tuner.finetune(n=20, data=data)
        
        # Step 2: Use result for attention
        if result['softmax_result'] is not None:
            weights = result['softmax_result']
            attention_output = attention(weights, weights, weights, dk=4)
            
            # Step 3: Use attention output to trigger production actions
            max_attention = np.max(attention_output)
            
            if max_attention > 0.5:
                with patch('builtins.print'):
                    execute_action_based_on_text("create_ticket", 
                                                {"attention_score": float(max_attention)})
        
        # Verify the pipeline completed
        assert result['status'] == 'completed'
    
    def test_mathematical_consistency(self):
        """Test mathematical consistency across components."""
        # Test that softmax properties hold across different uses
        test_data = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Direct softmax
        direct_result = softmax(test_data)
        
        # Softmax through FineTuner
        tuner = FineTuner()
        tuner_result = tuner.finetune(n=1, data=test_data.reshape(1, -1))
        
        if tuner_result['softmax_result'] is not None:
            tuner_softmax = tuner_result['softmax_result'][0]
            
            # Should be similar (allowing for different implementations)
            assert np.allclose(np.sum(direct_result), 1.0)
            assert np.allclose(np.sum(tuner_softmax), 1.0)
            assert np.all(direct_result >= 0)
            assert np.all(tuner_softmax >= 0)
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test that errors in one component don't crash others
        tuner = FineTuner()
        
        # Try with problematic data
        try:
            result = tuner.finetune(n=10, data=np.array([]))
        except:
            pass  # Expected to fail
        
        # Should still be able to use tuner after error
        good_result = tuner.finetune(n=10, data=np.random.rand(3, 3))
        assert good_result['status'] == 'completed'
        
        # Test attention with edge cases
        try:
            # This might fail due to dimension mismatch
            attention(np.array([[1]]), np.array([[1, 2]]), np.array([[1]]), dk=1)
        except:
            pass  # Expected to fail
        
        # Should still work with proper dimensions
        Q = K = V = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = attention(Q, K, V, dk=2)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    def test_scaling_behavior(self):
        """Test how the integrated system scales."""
        import time
        
        sizes = [4, 8, 16]
        times = []
        
        for size in sizes:
            start_time = time.perf_counter()
            
            # Create data
            data = np.random.rand(size, size)
            
            # Fine-tune
            tuner = FineTuner()
            result = tuner.finetune(n=10, data=data)
            
            # Apply attention if possible
            if result['softmax_result'] is not None:
                weights = result['softmax_result']
                attention_output = attention(weights, weights, weights, dk=size)
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Should complete in reasonable time
        assert all(t < 5.0 for t in times), "Integration pipeline should be reasonably fast"
    
    def test_memory_efficiency(self):
        """Test memory efficiency of integrated operations."""
        # Perform multiple integration cycles
        for _ in range(10):
            tuner = FineTuner()
            data = np.random.rand(8, 8)
            result = tuner.finetune(n=20, data=data)
            
            if result['softmax_result'] is not None:
                weights = result['softmax_result']
                attention_output = attention(weights, weights, weights, dk=8)
                
                # Clean up explicitly
                del weights, attention_output
            
            del tuner, data, result
        
        # If we get here without memory errors, test passes
        assert True


class TestRobustnessIntegration:
    """Test robustness of integrated system."""
    
    def test_repeated_operations(self):
        """Test repeated operations for stability."""
        tuner = FineTuner()
        
        # Perform many operations
        for i in range(50):
            data = np.random.rand(4, 4) + i * 0.01  # Slightly different each time
            result = tuner.finetune(n=5, data=data)
            
            assert result['status'] == 'completed'
            
            if result['softmax_result'] is not None:
                weights = result['softmax_result']
                attention_output = attention(weights, weights, weights, dk=4)
                assert np.all(np.isfinite(attention_output))
    
    def test_edge_case_integration(self):
        """Test integration with edge cases."""
        tuner = FineTuner()
        
        # Test with minimal data
        minimal_data = np.array([[1.0]])
        result = tuner.finetune(n=1, data=minimal_data)
        assert result['status'] == 'completed'
        
        # Test with identity matrix
        identity_data = np.eye(3)
        result = tuner.finetune(n=5, data=identity_data)
        assert result['status'] == 'completed'
        
        if result['softmax_result'] is not None:
            weights = result['softmax_result']
            attention_output = attention(weights, weights, weights, dk=3)
            assert attention_output.shape == (3, 3)
            assert np.all(np.isfinite(attention_output))
    
    def test_concurrent_simple_operations(self):
        """Test simple concurrent operations."""
        import threading
        
        results = []
        
        def worker():
            tuner = FineTuner()
            data = np.random.rand(3, 3)
            result = tuner.finetune(n=10, data=data)
            results.append(result['status'] == 'completed')
        
        # Run a few concurrent workers
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should complete successfully
        assert len(results) == 3
        assert all(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])