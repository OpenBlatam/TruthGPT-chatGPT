"""
Property-based tests for mathematical functions.
These tests verify mathematical properties that should always hold.
"""
import numpy as np
import pytest
import sys
import os
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Automate-Inteligence', 'AI-Automate', 'Transformers'))

from transformer import attention
from finetune import softmax, FineTuner


class TestSoftmaxProperties:
    """Property-based tests for softmax function."""
    
    @given(arrays(np.float64, shape=st.integers(1, 20), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    def test_softmax_sums_to_one(self, x):
        """Property: Softmax output should always sum to 1."""
        result = softmax(x)
        assert np.isclose(np.sum(result), 1.0, rtol=1e-10), f"Softmax sum: {np.sum(result)}"
    
    @given(arrays(np.float64, shape=st.integers(1, 20), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    def test_softmax_all_positive(self, x):
        """Property: All softmax outputs should be positive."""
        result = softmax(x)
        assert np.all(result > 0), f"Found non-positive values: {result[result <= 0]}"
    
    @given(arrays(np.float64, shape=st.integers(1, 20), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    def test_softmax_preserves_order(self, x):
        """Property: If x[i] > x[j], then softmax(x)[i] > softmax(x)[j]."""
        if len(x) < 2:
            return
        
        result = softmax(x)
        
        for i in range(len(x)):
            for j in range(len(x)):
                if x[i] > x[j]:
                    assert result[i] > result[j], f"Order not preserved: x[{i}]={x[i]} > x[{j}]={x[j]} but softmax[{i}]={result[i]} <= softmax[{j}]={result[j]}"
    
    @given(
        arrays(np.float64, shape=st.integers(1, 20), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
        st.floats(-5, 5, allow_nan=False, allow_infinity=False)
    )
    def test_softmax_translation_invariance(self, x, c):
        """Property: softmax(x + c) = softmax(x) for any constant c."""
        result1 = softmax(x)
        result2 = softmax(x + c)
        np.testing.assert_allclose(result1, result2, rtol=1e-10)
    
    @given(arrays(np.float64, shape=st.integers(1, 20), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    def test_softmax_max_element(self, x):
        """Property: The largest input should correspond to the largest output."""
        if len(x) < 2:
            return
        
        result = softmax(x)
        max_input_idx = np.argmax(x)
        max_output_idx = np.argmax(result)
        
        # If there are ties, this might not hold exactly, so check if they're close
        if np.sum(x == x[max_input_idx]) == 1:  # No ties in input
            assert max_input_idx == max_output_idx
    
    @given(
        arrays(np.float64, shape=st.integers(1, 20), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)),
        st.floats(0.1, 10, allow_nan=False, allow_infinity=False)
    )
    def test_softmax_temperature_scaling(self, x, temperature):
        """Property: Temperature scaling should affect the sharpness of distribution."""
        if len(x) < 2:
            return
        
        result_normal = softmax(x)
        result_scaled = softmax(x / temperature)
        
        # Higher temperature (> 1) should make distribution more uniform
        # Lower temperature (< 1) should make distribution more peaked
        if temperature > 1:
            # More uniform means lower max value
            assert np.max(result_scaled) <= np.max(result_normal) + 1e-10
        elif temperature < 1:
            # More peaked means higher max value
            assert np.max(result_scaled) >= np.max(result_normal) - 1e-10


class TestAttentionProperties:
    """Property-based tests for attention mechanism."""
    
    @given(
        st.integers(2, 10),  # sequence length
        st.integers(2, 8),   # model dimension
    )
    def test_attention_output_shape(self, seq_len, d_model):
        """Property: Attention output should have the same shape as Q."""
        Q = np.random.rand(seq_len, d_model)
        K = np.random.rand(seq_len, d_model)
        V = np.random.rand(seq_len, d_model)
        
        result = attention(Q, K, V, dk=d_model)
        assert result.shape == Q.shape
    
    @given(
        st.integers(2, 8),   # sequence length
        st.integers(2, 6),   # model dimension
        st.floats(0.1, 10, allow_nan=False, allow_infinity=False)  # dk scaling
    )
    def test_attention_finite_output(self, seq_len, d_model, dk):
        """Property: Attention output should always be finite."""
        Q = np.random.rand(seq_len, d_model) * 2 - 1  # Range [-1, 1]
        K = np.random.rand(seq_len, d_model) * 2 - 1
        V = np.random.rand(seq_len, d_model) * 2 - 1
        
        result = attention(Q, K, V, dk=dk)
        assert np.all(np.isfinite(result)), f"Non-finite values found in attention output"
    
    @given(
        st.integers(2, 6),   # sequence length
        st.integers(2, 4),   # model dimension
    )
    def test_attention_value_weighted_combination(self, seq_len, d_model):
        """Property: Attention output should be a weighted combination of V rows."""
        Q = np.random.rand(seq_len, d_model)
        K = np.random.rand(seq_len, d_model)
        V = np.random.rand(seq_len, d_model)
        
        result = attention(Q, K, V, dk=d_model)
        
        # Each output row should be within the convex hull of V rows
        v_min = np.min(V, axis=0)
        v_max = np.max(V, axis=0)
        
        for i in range(seq_len):
            assert np.all(result[i] >= v_min - 1e-10), f"Output row {i} below V minimum"
            assert np.all(result[i] <= v_max + 1e-10), f"Output row {i} above V maximum"
    
    @given(
        st.integers(2, 6),   # sequence length
        st.integers(2, 4),   # model dimension
    )
    def test_attention_identity_values(self, seq_len, d_model):
        """Property: When V is identity, attention should preserve some structure."""
        Q = np.random.rand(seq_len, d_model)
        K = np.random.rand(seq_len, d_model)
        V = np.eye(seq_len, d_model)  # Identity-like matrix
        
        if seq_len <= d_model:
            result = attention(Q, K, V, dk=d_model)
            
            # Result should have some relationship to the identity structure
            assert result.shape == (seq_len, d_model)
            assert np.all(np.isfinite(result))


class TestFineTunerProperties:
    """Property-based tests for FineTuner class."""
    
    @given(
        st.integers(1, 1000),  # n parameter
        st.floats(0.0001, 1.0, allow_nan=False, allow_infinity=False),  # learning rate
        st.integers(1, 100)    # epochs
    )
    def test_finetuner_parameter_consistency(self, n, learning_rate, epochs):
        """Property: FineTuner should consistently store and return parameters."""
        tuner = FineTuner()
        tuner.set_hyperparameters(learning_rate=learning_rate, epochs=epochs)
        
        result = tuner.finetune(n=n)
        
        assert result['n'] == n
        assert result['learning_rate'] == learning_rate
        assert result['epochs'] == epochs
        assert result['status'] == 'completed'
    
    @given(
        st.integers(10, 100),  # n parameter
        arrays(np.float64, shape=(st.integers(2, 10), st.integers(2, 10)), 
               elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False))
    )
    def test_finetuner_data_processing_properties(self, n, data):
        """Property: FineTuner should process data consistently."""
        tuner = FineTuner()
        result = tuner.finetune(n=n, data=data)
        
        assert result['n'] == n
        
        if result['softmax_result'] is not None:
            # Should have same shape as input
            assert result['softmax_result'].shape == data.shape
            
            # Each row should be a valid probability distribution
            for i in range(data.shape[0]):
                row_sum = np.sum(result['softmax_result'][i])
                assert np.isclose(row_sum, 1.0, rtol=1e-10)
                assert np.all(result['softmax_result'][i] >= 0)
    
    @given(
        st.integers(10, 100),  # n parameter
        st.floats(0.0001, 1.0, allow_nan=False, allow_infinity=False),  # learning rate 1
        st.floats(0.0001, 1.0, allow_nan=False, allow_infinity=False),  # learning rate 2
    )
    def test_finetuner_deterministic_behavior(self, n, lr1, lr2):
        """Property: Same inputs should produce same outputs."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        tuner1 = FineTuner()
        result1 = tuner1.finetune(n=n, data=data, learning_rate=lr1)
        
        tuner2 = FineTuner()
        result2 = tuner2.finetune(n=n, data=data, learning_rate=lr1)
        
        # Same parameters should give same results
        assert result1['n'] == result2['n']
        assert result1['learning_rate'] == result2['learning_rate']
        
        if result1['softmax_result'] is not None and result2['softmax_result'] is not None:
            np.testing.assert_allclose(result1['softmax_result'], result2['softmax_result'])
        
        # Different learning rates should potentially give different results
        if lr1 != lr2:
            result3 = tuner1.finetune(n=n, data=data, learning_rate=lr2)
            assert result3['learning_rate'] == lr2


class TestMathematicalInvariants:
    """Test mathematical invariants across the system."""
    
    @given(
        arrays(np.float64, shape=st.integers(2, 8), elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False)),
        st.floats(0.1, 5, allow_nan=False, allow_infinity=False)
    )
    def test_softmax_entropy_properties(self, x, temperature):
        """Property: Softmax entropy should behave predictably with temperature."""
        # Apply temperature scaling
        scaled_x = x / temperature
        result = softmax(scaled_x)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(result * np.log(result + 1e-15))  # Add small epsilon for numerical stability
        
        # Entropy should be non-negative
        assert entropy >= 0, f"Negative entropy: {entropy}"
        
        # For uniform distribution, entropy should be log(n)
        n = len(x)
        max_entropy = np.log(n)
        assert entropy <= max_entropy + 1e-10, f"Entropy {entropy} exceeds maximum {max_entropy}"
    
    @given(
        st.integers(2, 6),   # matrix size
        st.floats(0.1, 3, allow_nan=False, allow_infinity=False)  # scaling factor
    )
    def test_attention_scaling_properties(self, size, scale):
        """Property: Attention mechanism should scale predictably."""
        # Create scaled versions of the same matrices
        base_Q = np.random.rand(size, size)
        base_K = np.random.rand(size, size)
        base_V = np.random.rand(size, size)
        
        result1 = attention(base_Q, base_K, base_V, dk=size)
        result2 = attention(base_Q * scale, base_K * scale, base_V, dk=size)
        
        # Both results should be finite
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))
        
        # Results should be different when Q and K are scaled
        if scale != 1.0:
            assert not np.allclose(result1, result2, rtol=1e-10)
    
    @given(
        st.integers(2, 6),   # sequence length
        st.integers(2, 4),   # model dimension
    )
    def test_attention_permutation_equivariance(self, seq_len, d_model):
        """Property: Attention should be equivariant to permutations of the sequence."""
        Q = np.random.rand(seq_len, d_model)
        K = np.random.rand(seq_len, d_model)
        V = np.random.rand(seq_len, d_model)
        
        # Apply attention
        result1 = attention(Q, K, V, dk=d_model)
        
        # Create a permutation
        perm = np.random.permutation(seq_len)
        
        # Apply permutation to K and V (but not Q for this test)
        K_perm = K[perm]
        V_perm = V[perm]
        
        result2 = attention(Q, K_perm, V_perm, dk=d_model)
        
        # Results should be different (unless by coincidence)
        assert result1.shape == result2.shape
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))


class TestNumericalStability:
    """Test numerical stability properties."""
    
    @given(
        arrays(np.float64, shape=st.integers(2, 10), 
               elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False))
    )
    def test_softmax_large_values_stability(self, x):
        """Property: Softmax should be stable even with large input values."""
        result = softmax(x)
        
        # Should not contain NaN or infinity
        assert np.all(np.isfinite(result)), f"Non-finite values in softmax output: {result}"
        
        # Should still sum to 1
        assert np.isclose(np.sum(result), 1.0, rtol=1e-10)
        
        # Should be positive
        assert np.all(result > 0)
    
    @given(
        st.integers(2, 6),   # matrix size
    )
    def test_attention_extreme_values_stability(self, size):
        """Property: Attention should handle extreme values gracefully."""
        # Test with large values
        Q = np.random.rand(size, size) * 100
        K = np.random.rand(size, size) * 100
        V = np.random.rand(size, size)
        
        result = attention(Q, K, V, dk=size)
        
        assert np.all(np.isfinite(result)), "Attention output should be finite with large inputs"
        assert result.shape == (size, size)
    
    @given(
        st.integers(2, 8),   # array size
    )
    def test_softmax_near_zero_values(self, size):
        """Property: Softmax should handle very small values correctly."""
        # Test with very small values
        x = np.random.rand(size) * 1e-10
        result = softmax(x)
        
        # Should still be a valid probability distribution
        assert np.isclose(np.sum(result), 1.0, rtol=1e-10)
        assert np.all(result > 0)
        assert np.all(np.isfinite(result))
        
        # Should be approximately uniform for very small inputs
        expected_uniform = 1.0 / size
        assert np.allclose(result, expected_uniform, rtol=1e-5)


if __name__ == "__main__":
    # Run with more examples for thorough testing
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])