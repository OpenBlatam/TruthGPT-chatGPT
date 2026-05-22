"""
Unit tests for the finetune module.
"""
import numpy as np
import pytest
import sys
import os

# Add the current directory to import the finetune module
sys.path.append(os.path.dirname(__file__))

from finetune import softmax, FineTuner, finetune


class TestSoftmax:
    """Test cases for the softmax function."""
    
    def test_softmax_basic(self):
        """Test basic softmax functionality."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        
        # Check that result sums to 1
        assert np.isclose(np.sum(result), 1.0), "Softmax output should sum to 1"
        
        # Check that all values are positive
        assert np.all(result > 0), "All softmax values should be positive"
        
        # Check that result has same shape as input
        assert result.shape == x.shape, "Output shape should match input shape"
    
    def test_softmax_numerical_stability(self):
        """Test softmax with large values for numerical stability."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        
        # Should not contain NaN or infinite values
        assert np.all(np.isfinite(result)), "Softmax should be numerically stable"
        assert np.isclose(np.sum(result), 1.0), "Softmax output should sum to 1"
    
    def test_softmax_uniform_input(self):
        """Test softmax with uniform input values."""
        x = np.array([5.0, 5.0, 5.0])
        result = softmax(x)
        
        # All values should be equal (approximately 1/3)
        expected = 1.0 / 3.0
        np.testing.assert_allclose(result, [expected, expected, expected], rtol=1e-10)
    
    def test_softmax_single_element(self):
        """Test softmax with single element."""
        x = np.array([42.0])
        result = softmax(x)
        
        # Single element should give [1.0]
        np.testing.assert_allclose(result, [1.0])
    
    def test_softmax_negative_values(self):
        """Test softmax with negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        result = softmax(x)
        
        # Should still sum to 1 and be positive
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(result > 0)
    
    def test_softmax_zero_input(self):
        """Test softmax with zero input."""
        x = np.array([0.0, 0.0, 0.0])
        result = softmax(x)
        
        # All values should be equal (1/3)
        expected = 1.0 / 3.0
        np.testing.assert_allclose(result, [expected, expected, expected])


class TestFineTuner:
    """Test cases for the FineTuner class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.tuner = FineTuner()
    
    def test_finetuner_initialization(self):
        """Test FineTuner initialization."""
        assert self.tuner.n is None
        assert self.tuner.learning_rate == 0.001
        assert self.tuner.epochs == 10
    
    def test_finetune_basic(self):
        """Test basic fine-tuning functionality."""
        result = self.tuner.finetune(n=100)
        
        assert result['n'] == 100
        assert result['learning_rate'] == 0.001
        assert result['epochs'] == 10
        assert result['status'] == 'completed'
        assert result['softmax_result'] is None
    
    def test_finetune_with_data_1d(self):
        """Test fine-tuning with 1D data."""
        data = [1.0, 2.0, 3.0]
        result = self.tuner.finetune(n=50, data=data)
        
        assert result['n'] == 50
        assert result['softmax_result'] is not None
        
        # Check that softmax was applied correctly
        expected_softmax = softmax(np.array(data))
        np.testing.assert_allclose(result['softmax_result'], expected_softmax)
    
    def test_finetune_with_data_2d(self):
        """Test fine-tuning with 2D data."""
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = self.tuner.finetune(n=25, data=data)
        
        assert result['n'] == 25
        assert result['softmax_result'] is not None
        assert result['softmax_result'].shape == (3, 2)
        
        # Check that softmax was applied to each row
        for i, row in enumerate(data):
            expected_row_softmax = softmax(np.array(row))
            np.testing.assert_allclose(result['softmax_result'][i], expected_row_softmax)
    
    def test_finetune_with_numpy_array(self):
        """Test fine-tuning with numpy array data."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = self.tuner.finetune(n=75, data=data)
        
        assert result['n'] == 75
        assert result['softmax_result'] is not None
        
        expected_softmax = softmax(data)
        np.testing.assert_allclose(result['softmax_result'], expected_softmax)
    
    def test_finetune_with_custom_learning_rate(self):
        """Test fine-tuning with custom learning rate."""
        result = self.tuner.finetune(n=30, learning_rate=0.01)
        
        assert result['learning_rate'] == 0.01
        assert self.tuner.learning_rate == 0.01
    
    def test_finetune_with_invalid_data(self):
        """Test fine-tuning with invalid data type."""
        result = self.tuner.finetune(n=40, data="invalid_data")
        
        assert result['n'] == 40
        assert result['softmax_result'] is None
    
    def test_set_hyperparameters(self):
        """Test setting hyperparameters."""
        self.tuner.set_hyperparameters(learning_rate=0.05, epochs=20)
        
        assert self.tuner.learning_rate == 0.05
        assert self.tuner.epochs == 20
    
    def test_set_partial_hyperparameters(self):
        """Test setting only some hyperparameters."""
        original_epochs = self.tuner.epochs
        
        self.tuner.set_hyperparameters(learning_rate=0.02)
        
        assert self.tuner.learning_rate == 0.02
        assert self.tuner.epochs == original_epochs  # Should remain unchanged
    
    def test_get_hyperparameters(self):
        """Test getting hyperparameters."""
        self.tuner.set_hyperparameters(learning_rate=0.03, epochs=15)
        
        params = self.tuner.get_hyperparameters()
        
        assert params['learning_rate'] == 0.03
        assert params['epochs'] == 15
    
    def test_multiple_finetune_calls(self):
        """Test multiple fine-tuning calls."""
        # First call
        result1 = self.tuner.finetune(n=10, learning_rate=0.01)
        assert result1['n'] == 10
        assert result1['learning_rate'] == 0.01
        
        # Second call with different parameters
        result2 = self.tuner.finetune(n=20, learning_rate=0.02)
        assert result2['n'] == 20
        assert result2['learning_rate'] == 0.02
        
        # Check that tuner state was updated
        assert self.tuner.n == 20
        assert self.tuner.learning_rate == 0.02


class TestStandaloneFinetuneFunction:
    """Test cases for the standalone finetune function."""
    
    def test_standalone_finetune_basic(self):
        """Test basic standalone fine-tune function."""
        result = finetune(n=100)
        
        assert result['n'] == 100
        assert result['learning_rate'] == 0.001
        assert result['status'] == 'completed'
    
    def test_standalone_finetune_with_data(self):
        """Test standalone fine-tune function with data."""
        data = [2.0, 4.0, 6.0]
        result = finetune(n=50, data=data, learning_rate=0.005)
        
        assert result['n'] == 50
        assert result['learning_rate'] == 0.005
        assert result['softmax_result'] is not None
        
        expected_softmax = softmax(np.array(data))
        np.testing.assert_allclose(result['softmax_result'], expected_softmax)
    
    def test_standalone_finetune_with_custom_learning_rate(self):
        """Test standalone fine-tune function with custom learning rate."""
        result = finetune(n=25, learning_rate=0.1)
        
        assert result['n'] == 25
        assert result['learning_rate'] == 0.1
    
    def test_standalone_finetune_empty_data(self):
        """Test standalone fine-tune function with empty data."""
        result = finetune(n=15, data=[])
        
        assert result['n'] == 15
        # Empty list should be handled gracefully
        assert result['softmax_result'] is not None
        assert len(result['softmax_result']) == 0


class TestIntegration:
    """Integration tests for the finetune module."""
    
    def test_softmax_integration_with_finetune(self):
        """Test that softmax is correctly integrated with fine-tuning."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test with FineTuner class
        tuner = FineTuner()
        result_class = tuner.finetune(n=100, data=data)
        
        # Test with standalone function
        result_function = finetune(n=100, data=data)
        
        # Both should produce the same softmax result
        np.testing.assert_allclose(
            result_class['softmax_result'], 
            result_function['softmax_result']
        )
    
    def test_hyperparameter_persistence(self):
        """Test that hyperparameters persist across operations."""
        tuner = FineTuner()
        tuner.set_hyperparameters(learning_rate=0.05, epochs=25)
        
        # Fine-tune without specifying learning rate
        result = tuner.finetune(n=50)
        
        # Should use the previously set learning rate
        assert result['learning_rate'] == 0.05
        assert result['epochs'] == 25
    
    def test_large_data_handling(self):
        """Test handling of larger datasets."""
        # Create a larger 2D dataset
        data = np.random.rand(100, 10)
        
        result = finetune(n=1000, data=data)
        
        assert result['n'] == 1000
        assert result['softmax_result'].shape == (100, 10)
        
        # Check that each row sums to 1 (softmax property)
        row_sums = np.sum(result['softmax_result'], axis=1)
        np.testing.assert_allclose(row_sums, np.ones(100), rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])