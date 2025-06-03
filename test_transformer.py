"""
Unit tests for the transformer attention mechanism.
"""
import numpy as np
import pytest
import sys
import os

# Add the path to import the transformer module
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'Transformers'))

from transformer import attention


class TestAttentionMechanism:
    """Test cases for the attention function."""
    
    def test_attention_basic_functionality(self):
        """Test that attention function works with basic inputs."""
        # Create simple test matrices
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[2.0, 3.0], [4.0, 5.0]])
        dk = 2
        
        result = attention(Q, K, V, dk)
        
        # Check that result has correct shape
        assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
        
        # Check that result contains finite values
        assert np.all(np.isfinite(result)), "Result contains non-finite values"
    
    def test_attention_output_shape(self):
        """Test that attention output has correct shape for different input sizes."""
        # Test with different matrix sizes
        test_cases = [
            (2, 2),
            (3, 3),
            (4, 2),
            (2, 4)
        ]
        
        for seq_len, d_model in test_cases:
            Q = np.random.rand(seq_len, d_model)
            K = np.random.rand(seq_len, d_model)
            V = np.random.rand(seq_len, d_model)
            dk = d_model
            
            result = attention(Q, K, V, dk)
            expected_shape = (seq_len, d_model)
            
            assert result.shape == expected_shape, \
                f"For input shape ({seq_len}, {d_model}), expected output shape {expected_shape}, got {result.shape}"
    
    def test_attention_with_zeros(self):
        """Test attention mechanism with zero matrices."""
        Q = np.zeros((2, 2))
        K = np.zeros((2, 2))
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        dk = 2
        
        result = attention(Q, K, V, dk)
        
        # With zero Q and K, the attention weights should be uniform (0.5, 0.5)
        # So the result should be the average of V rows
        expected = np.array([[2.0, 3.0], [2.0, 3.0]])  # Average of V rows
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    
    def test_attention_identity_case(self):
        """Test attention with identity matrices."""
        Q = np.eye(2)
        K = np.eye(2)
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        dk = 2
        
        result = attention(Q, K, V, dk)
        
        # With identity Q and K, each query should attend most to itself
        # Check that result is finite and has correct shape
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))
    
    def test_attention_scaling_effect(self):
        """Test that dk scaling parameter affects the output."""
        # Use matrices that will produce different attention patterns with different scaling
        Q = np.array([[2.0, 0.0], [0.0, 2.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # Test with different dk values
        result_dk_1 = attention(Q, K, V, dk=1)
        result_dk_2 = attention(Q, K, V, dk=2)
        result_dk_4 = attention(Q, K, V, dk=4)
        
        # Results should be different due to scaling
        assert not np.allclose(result_dk_1, result_dk_2, rtol=1e-5), "Results should differ with different dk values"
        assert not np.allclose(result_dk_2, result_dk_4, rtol=1e-5), "Results should differ with different dk values"
    
    def test_attention_numerical_stability(self):
        """Test attention with large values to check numerical stability."""
        # Create matrices with large values
        Q = np.array([[100.0, 0.0], [0.0, 100.0]])
        K = np.array([[100.0, 0.0], [0.0, 100.0]])
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        dk = 2
        
        result = attention(Q, K, V, dk)
        
        # Check that result doesn't contain NaN or infinite values
        assert np.all(np.isfinite(result)), "Result should be finite even with large inputs"
        assert not np.any(np.isnan(result)), "Result should not contain NaN values"
    
    def test_attention_value_preservation(self):
        """Test that attention preserves the value space."""
        Q = np.random.rand(3, 2)
        K = np.random.rand(3, 2)
        V = np.random.rand(3, 2)
        dk = 2
        
        result = attention(Q, K, V, dk)
        
        # The result should be a weighted combination of V rows
        # So each column's values should be within the range of V's column values
        for col in range(V.shape[1]):
            v_min, v_max = V[:, col].min(), V[:, col].max()
            result_min, result_max = result[:, col].min(), result[:, col].max()
            
            # Allow for small numerical errors
            assert result_min >= v_min - 1e-10, f"Result minimum {result_min} below V minimum {v_min}"
            assert result_max <= v_max + 1e-10, f"Result maximum {result_max} above V maximum {v_max}"
    
    def test_attention_with_invalid_dk(self):
        """Test attention behavior with edge case dk values."""
        Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = np.array([[1.0, 0.0], [0.0, 1.0]])
        V = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Test with very small dk
        result_small_dk = attention(Q, K, V, dk=0.001)
        assert np.all(np.isfinite(result_small_dk)), "Should handle very small dk values"
        
        # Test with very large dk
        result_large_dk = attention(Q, K, V, dk=1000)
        assert np.all(np.isfinite(result_large_dk)), "Should handle very large dk values"


if __name__ == "__main__":
    pytest.main([__file__])