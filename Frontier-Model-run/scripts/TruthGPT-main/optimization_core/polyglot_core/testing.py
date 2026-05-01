"""
Testing utilities for polyglot_core.

Provides fixtures, helpers, and utilities for testing.
"""

from typing import Any, Optional, Callable, Dict
import numpy as np
import pytest


class PolyglotTestFixtures:
    """
    Test fixtures for polyglot_core.
    
    Provides common test data and utilities.
    """
    
    @staticmethod
    def create_test_tensor(shape: tuple, dtype: np.dtype = np.float32, seed: int = 42) -> np.ndarray:
        """
        Create test tensor.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            seed: Random seed
            
        Returns:
            NumPy array
        """
        np.random.seed(seed)
        return np.random.randn(*shape).astype(dtype)
    
    @staticmethod
    def create_test_cache_data(layer: int = 0, position: int = 0, dim: int = 64) -> tuple:
        """
        Create test cache data.
        
        Args:
            layer: Layer index
            position: Position index
            dim: Dimension size
            
        Returns:
            Tuple of (key, value)
        """
        np.random.seed(42)
        key = np.random.randn(dim).astype(np.float32)
        value = np.random.randn(dim).astype(np.float32)
        return key, value
    
    @staticmethod
    def create_test_attention_inputs(
        batch_size: int = 2,
        seq_len: int = 8,
        d_model: int = 256
    ) -> tuple:
        """
        Create test attention inputs.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            d_model: Model dimension
            
        Returns:
            Tuple of (query, key, value)
        """
        np.random.seed(42)
        q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
        k = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
        v = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
        return q, k, v
    
    @staticmethod
    def create_test_compression_data(size: int = 1000) -> bytes:
        """
        Create test compression data.
        
        Args:
            size: Data size in bytes
            
        Returns:
            Test data bytes
        """
        return b"Test compression data " * (size // 22 + 1)


@pytest.fixture
def test_tensor():
    """Pytest fixture for test tensor."""
    return PolyglotTestFixtures.create_test_tensor((10, 20))


@pytest.fixture
def test_cache_data():
    """Pytest fixture for test cache data."""
    return PolyglotTestFixtures.create_test_cache_data()


@pytest.fixture
def test_attention_inputs():
    """Pytest fixture for test attention inputs."""
    return PolyglotTestFixtures.create_test_attention_inputs()


@pytest.fixture
def test_compression_data():
    """Pytest fixture for test compression data."""
    return PolyglotTestFixtures.create_test_compression_data()


class TestHelpers:
    """Helper functions for testing."""
    
    @staticmethod
    def assert_tensor_equal(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-5):
        """Assert tensors are equal."""
        np.testing.assert_allclose(actual, expected, rtol=rtol)
    
    @staticmethod
    def assert_tensor_shape(tensor: np.ndarray, expected_shape: tuple):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    @staticmethod
    def assert_backend_available(backend_name: str):
        """Assert backend is available."""
        try:
            from .backend import is_backend_available
            assert is_backend_available(backend_name), f"Backend {backend_name} is not available"
        except ImportError:
            pytest.skip(f"Backend module not available")
    
    @staticmethod
    def skip_if_backend_unavailable(backend_name: str):
        """Skip test if backend is unavailable."""
        try:
            from .backend import is_backend_available
            if not is_backend_available(backend_name):
                pytest.skip(f"Backend {backend_name} is not available")
        except ImportError:
            pytest.skip(f"Backend module not available")
    
    @staticmethod
    def create_mock_cache(max_size: int = 100):
        """Create mock cache for testing."""
        from .cache import KVCache
        return KVCache(max_size=max_size)
    
    @staticmethod
    def create_mock_attention(d_model: int = 256, n_heads: int = 4):
        """Create mock attention for testing."""
        from .attention import Attention, AttentionConfig
        config = AttentionConfig(d_model=d_model, n_heads=n_heads)
        return Attention(config)
    
    @staticmethod
    def create_mock_compressor(algorithm: str = "lz4"):
        """Create mock compressor for testing."""
        from .compression import Compressor
        return Compressor(algorithm=algorithm)


def assert_tensor_equal(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-5):
    """Convenience function to assert tensors are equal."""
    TestHelpers.assert_tensor_equal(actual, expected, rtol)


def assert_tensor_shape(tensor: np.ndarray, expected_shape: tuple):
    """Convenience function to assert tensor shape."""
    TestHelpers.assert_tensor_shape(tensor, expected_shape)













