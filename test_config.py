"""
Configuration and utility tests for the TruthGPT system.
Tests configuration management, error handling, and edge cases.
"""
import numpy as np
import pytest
import sys
import os
import tempfile
import json
import pickle
from unittest.mock import Mock, patch, MagicMock
import warnings

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'Transformers'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture', 'NLP'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'MAIN-Architecture'))

from transformer import attention
from manager import CommandManager
from production import execute_action_based_on_text
from finetune import FineTuner, softmax


class TestConfigurationManagement:
    """Test configuration and parameter management."""
    
    def test_finetuner_default_configuration(self):
        """Test FineTuner default configuration."""
        tuner = FineTuner()
        
        # Test default hyperparameters
        result = tuner.finetune(n=10)
        
        assert 'learning_rate' in result
        assert 'epochs' in result
        assert result['learning_rate'] > 0
        assert result['epochs'] > 0
        assert isinstance(result['learning_rate'], float)
        assert isinstance(result['epochs'], int)
    
    def test_finetuner_configuration_persistence(self):
        """Test that FineTuner configuration persists across calls."""
        tuner = FineTuner()
        
        # Set custom hyperparameters
        custom_lr = 0.123
        custom_epochs = 42
        tuner.set_hyperparameters(learning_rate=custom_lr, epochs=custom_epochs)
        
        # First call
        result1 = tuner.finetune(n=10)
        assert result1['learning_rate'] == custom_lr
        assert result1['epochs'] == custom_epochs
        
        # Second call should maintain the same configuration
        result2 = tuner.finetune(n=20)
        assert result2['learning_rate'] == custom_lr
        assert result2['epochs'] == custom_epochs
    
    def test_finetuner_configuration_validation(self):
        """Test FineTuner configuration validation."""
        tuner = FineTuner()
        
        # Test invalid learning rates
        invalid_lrs = [-1.0, 0.0, float('inf'), float('nan')]
        for lr in invalid_lrs:
            with pytest.raises((ValueError, TypeError)):
                tuner.set_hyperparameters(learning_rate=lr)
        
        # Test invalid epochs
        invalid_epochs = [-1, 0, 1.5, "invalid"]
        for epochs in invalid_epochs:
            with pytest.raises((ValueError, TypeError)):
                tuner.set_hyperparameters(epochs=epochs)
    
    def test_command_manager_configuration(self):
        """Test CommandManager configuration options."""
        # Test different queue modes
        manager_fifo = CommandManager(mode='FIFO')
        manager_lifo = CommandManager(mode='LIFO')
        
        class TestCommand:
            def __init__(self, name):
                self.name = name
                self.executed = False
            
            def execute(self):
                self.executed = True
            
            def undo(self):
                self.executed = False
        
        # Add commands to both managers
        cmd1 = TestCommand("first")
        cmd2 = TestCommand("second")
        
        manager_fifo.add_command(cmd1)
        manager_fifo.add_command(cmd2)
        
        manager_lifo.add_command(TestCommand("first"))
        manager_lifo.add_command(TestCommand("second"))
        
        # Both should work regardless of mode
        manager_fifo.execute_commands()
        manager_lifo.execute_commands()
        
        assert cmd1.executed
        assert cmd2.executed


class TestErrorHandlingAndRecovery:
    """Test comprehensive error handling and recovery mechanisms."""
    
    def test_softmax_error_handling(self):
        """Test softmax error handling with invalid inputs."""
        # Test with None
        with pytest.raises((TypeError, AttributeError)):
            softmax(None)
        
        # Test with string
        with pytest.raises((TypeError, AttributeError)):
            softmax("invalid")
        
        # Test with list of strings
        with pytest.raises((TypeError, ValueError)):
            softmax(["a", "b", "c"])
        
        # Test with mixed types
        with pytest.raises((TypeError, ValueError)):
            softmax([1, "a", 3])
    
    def test_attention_error_handling(self):
        """Test attention mechanism error handling."""
        # Test with mismatched dimensions
        Q = np.random.rand(3, 4)
        K = np.random.rand(5, 4)  # Different number of rows
        V = np.random.rand(3, 4)
        
        with pytest.raises((ValueError, IndexError)):
            attention(Q, K, V, dk=4)
        
        # Test with incompatible dimensions
        Q = np.random.rand(3, 4)
        K = np.random.rand(3, 6)  # Different number of columns
        V = np.random.rand(3, 4)
        
        with pytest.raises((ValueError, IndexError)):
            attention(Q, K, V, dk=4)
        
        # Test with None inputs
        with pytest.raises((TypeError, AttributeError)):
            attention(None, K, V, dk=4)
        
        # Test with invalid dk
        Q = np.random.rand(3, 4)
        K = np.random.rand(3, 4)
        V = np.random.rand(3, 4)
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            attention(Q, K, V, dk=0)
    
    def test_finetuner_error_recovery(self):
        """Test FineTuner error recovery mechanisms."""
        tuner = FineTuner()
        
        # Test with invalid n parameter
        with pytest.raises((ValueError, TypeError)):
            tuner.finetune(n=-1)
        
        with pytest.raises((ValueError, TypeError)):
            tuner.finetune(n="invalid")
        
        # Test that tuner can recover after errors
        try:
            tuner.finetune(n=-1)
        except:
            pass
        
        # Should still work after error
        result = tuner.finetune(n=10)
        assert result['status'] == 'completed'
    
    def test_command_manager_error_recovery(self):
        """Test CommandManager error recovery."""
        manager = CommandManager()
        
        class FailingCommand:
            def __init__(self, should_fail=True):
                self.should_fail = should_fail
                self.executed = False
            
            def execute(self):
                if self.should_fail:
                    raise RuntimeError("Command failed")
                self.executed = True
            
            def undo(self):
                pass
        
        class SuccessfulCommand:
            def __init__(self):
                self.executed = False
            
            def execute(self):
                self.executed = True
            
            def undo(self):
                self.executed = False
        
        # Add failing and successful commands
        failing_cmd = FailingCommand()
        successful_cmd = SuccessfulCommand()
        
        manager.add_command(failing_cmd)
        manager.add_command(successful_cmd)
        
        # Execute commands - should handle failures gracefully
        manager.execute_commands()
        
        # Successful command should still execute despite failing command
        assert not failing_cmd.executed
        assert successful_cmd.executed
    
    def test_production_error_handling(self):
        """Test production system error handling."""
        # Test with invalid action
        with patch('builtins.print'):  # Suppress output
            result = execute_action_based_on_text("invalid_action", {})
            # Should not raise exception, just handle gracefully
        
        # Test with None parameters
        with patch('builtins.print'):
            result = execute_action_based_on_text("create_ticket", None)
        
        # Test with malformed parameters
        with patch('builtins.print'):
            result = execute_action_based_on_text("create_ticket", "invalid_params")


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""
    
    def test_softmax_edge_cases(self):
        """Test softmax with edge cases."""
        # Single element
        result = softmax(np.array([5.0]))
        assert np.isclose(result[0], 1.0)
        
        # Two identical elements
        result = softmax(np.array([2.0, 2.0]))
        assert np.allclose(result, [0.5, 0.5])
        
        # Very large difference
        result = softmax(np.array([0.0, 100.0]))
        assert result[0] < 1e-40  # Should be very small
        assert result[1] > 0.999  # Should be very close to 1
        
        # All zeros
        result = softmax(np.array([0.0, 0.0, 0.0]))
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(result, expected)
    
    def test_attention_edge_cases(self):
        """Test attention mechanism edge cases."""
        # Single token
        Q = np.array([[1.0]])
        K = np.array([[2.0]])
        V = np.array([[3.0]])
        
        result = attention(Q, K, V, dk=1)
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 3.0)  # Should return the value
        
        # Identity matrices
        size = 4
        I = np.eye(size)
        result = attention(I, I, I, dk=size)
        
        # Should be close to identity for this special case
        assert result.shape == (size, size)
        assert np.all(np.isfinite(result))
        
        # Zero matrices
        Z = np.zeros((3, 3))
        V = np.ones((3, 3))
        result = attention(Z, Z, V, dk=3)
        
        # Should produce finite output
        assert np.all(np.isfinite(result))
        assert result.shape == (3, 3)
    
    def test_finetuner_boundary_conditions(self):
        """Test FineTuner boundary conditions."""
        tuner = FineTuner()
        
        # Minimum n value
        result = tuner.finetune(n=1)
        assert result['n'] == 1
        assert result['status'] == 'completed'
        
        # Single data point
        data = np.array([[1.0]])
        result = tuner.finetune(n=10, data=data)
        assert result['status'] == 'completed'
        
        # Very small learning rate
        result = tuner.finetune(n=10, learning_rate=1e-10)
        assert result['learning_rate'] == 1e-10
        assert result['status'] == 'completed'
    
    def test_command_manager_boundary_conditions(self):
        """Test CommandManager boundary conditions."""
        manager = CommandManager()
        
        # Empty command execution
        manager.execute_commands()  # Should not crash
        
        # Undo with no commands
        manager.undo_last_command()  # Should not crash
        
        # Single command
        class SingleCommand:
            def __init__(self):
                self.executed = False
            
            def execute(self):
                self.executed = True
            
            def undo(self):
                self.executed = False
        
        cmd = SingleCommand()
        manager.add_command(cmd)
        manager.execute_commands()
        assert cmd.executed
        
        manager.undo_last_command()
        assert not cmd.executed


class TestDataValidationAndSanitization:
    """Test data validation and sanitization."""
    
    def test_input_data_validation(self):
        """Test input data validation across components."""
        # Test numpy array validation
        valid_arrays = [
            np.array([1, 2, 3]),
            np.array([[1, 2], [3, 4]]),
            np.random.rand(5, 5)
        ]
        
        for arr in valid_arrays:
            result = softmax(arr.flatten())
            assert np.all(np.isfinite(result))
            assert np.isclose(np.sum(result), 1.0)
    
    def test_parameter_sanitization(self):
        """Test parameter sanitization."""
        tuner = FineTuner()
        
        # Test parameter bounds
        result = tuner.finetune(n=10, learning_rate=0.5)
        assert 0 < result['learning_rate'] <= 1.0
        
        # Test integer conversion
        result = tuner.finetune(n=10.7)  # Float that should be converted
        assert isinstance(result['n'], int)
        assert result['n'] == 10
    
    def test_output_validation(self):
        """Test output validation."""
        tuner = FineTuner()
        data = np.random.rand(5, 5)
        
        result = tuner.finetune(n=50, data=data)
        
        # Validate output structure
        required_keys = ['n', 'learning_rate', 'epochs', 'status']
        for key in required_keys:
            assert key in result
        
        # Validate output types
        assert isinstance(result['n'], int)
        assert isinstance(result['learning_rate'], float)
        assert isinstance(result['epochs'], int)
        assert isinstance(result['status'], str)
        
        # Validate output values
        assert result['n'] > 0
        assert result['learning_rate'] > 0
        assert result['epochs'] > 0
        assert result['status'] in ['completed', 'failed', 'in_progress']


class TestCompatibilityAndVersioning:
    """Test compatibility and version handling."""
    
    def test_numpy_version_compatibility(self):
        """Test compatibility with different numpy operations."""
        # Test with different numpy dtypes
        dtypes = [np.float32, np.float64, np.int32, np.int64]
        
        for dtype in dtypes:
            try:
                x = np.array([1, 2, 3], dtype=dtype)
                result = softmax(x.astype(np.float64))  # Convert to float for softmax
                assert np.all(np.isfinite(result))
            except Exception as e:
                pytest.fail(f"Failed with dtype {dtype}: {e}")
    
    def test_backward_compatibility(self):
        """Test backward compatibility of interfaces."""
        # Test that old-style calls still work
        tuner = FineTuner()
        
        # Old-style call with positional arguments
        result = tuner.finetune(100)
        assert result['n'] == 100
        
        # Mixed positional and keyword arguments
        result = tuner.finetune(50, learning_rate=0.01)
        assert result['n'] == 50
        assert result['learning_rate'] == 0.01
    
    def test_future_compatibility(self):
        """Test future compatibility considerations."""
        # Test with additional parameters that might be added
        tuner = FineTuner()
        
        # Should handle unknown keyword arguments gracefully
        try:
            result = tuner.finetune(n=10, unknown_param="value")
            # If it doesn't raise an error, that's fine
        except TypeError:
            # If it raises TypeError for unknown params, that's also acceptable
            pass


class TestDocumentationAndExamples:
    """Test that documentation examples work correctly."""
    
    def test_basic_usage_examples(self):
        """Test basic usage examples that might be in documentation."""
        # Example 1: Basic softmax
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.isclose(np.sum(result), 1.0)
        assert len(result) == 3
        
        # Example 2: Basic attention
        Q = np.random.rand(4, 8)
        K = np.random.rand(4, 8)
        V = np.random.rand(4, 8)
        result = attention(Q, K, V, dk=8)
        assert result.shape == (4, 8)
        
        # Example 3: Basic fine-tuning
        tuner = FineTuner()
        result = tuner.finetune(n=100)
        assert result['status'] == 'completed'
        
        # Example 4: Basic command management
        manager = CommandManager()
        
        class ExampleCommand:
            def execute(self):
                pass
            def undo(self):
                pass
        
        manager.add_command(ExampleCommand())
        manager.execute_commands()
    
    def test_advanced_usage_examples(self):
        """Test advanced usage examples."""
        # Example: Multi-head attention simulation
        num_heads = 4
        seq_len = 8
        d_model = 32
        d_k = d_model // num_heads
        
        # Create random input
        input_data = np.random.rand(seq_len, d_model)
        
        # Simulate multi-head attention
        heads = []
        for i in range(num_heads):
            start = i * d_k
            end = start + d_k
            
            Q = input_data[:, start:end]
            K = input_data[:, start:end]
            V = input_data[:, start:end]
            
            head_output = attention(Q, K, V, dk=d_k)
            heads.append(head_output)
        
        # Concatenate heads
        multi_head_output = np.concatenate(heads, axis=1)
        assert multi_head_output.shape == (seq_len, d_model)
        
        # Example: Fine-tuning with custom data
        tuner = FineTuner()
        custom_data = np.random.rand(10, 16)
        result = tuner.finetune(n=200, data=custom_data, learning_rate=0.01)
        
        assert result['status'] == 'completed'
        if result['softmax_result'] is not None:
            assert result['softmax_result'].shape == custom_data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])