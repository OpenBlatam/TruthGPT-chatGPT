"""
Test fixtures for optimization_core tests.

Provides reusable fixtures for common test scenarios.
"""
import logging
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Test configuration fixture."""
    model_name: str = "test-model"
    model_path: str = "/tmp/test-model"
    batch_size: int = 8
    max_tokens: int = 64
    temperature: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "batch_size": self.batch_size,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


class MockInferenceEngine:
    """Mock inference engine for testing."""
    
    def __init__(self, config: Optional[TestConfig] = None):
        """Initialize mock engine."""
        self.config = config or TestConfig()
        self.is_initialized = True
        self._call_count = 0
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Mock generate method."""
        self._call_count += 1
        max_tokens = max_tokens or self.config.max_tokens
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        return [
            f"Generated text for: {prompt} (tokens={max_tokens})"
            for prompt in prompts
        ]
    
    def __call__(self, prompts, **kwargs):
        """Make engine callable."""
        return self.generate(prompts, **kwargs)
    
    @property
    def call_count(self) -> int:
        """Get number of calls."""
        return self._call_count


class MockDataProcessor:
    """Mock data processor for testing."""
    
    def __init__(self, lazy: bool = True):
        """Initialize mock processor."""
        self.lazy = lazy
        self._read_count = 0
    
    def read_parquet(self, path, **kwargs):
        """Mock read_parquet method."""
        self._read_count += 1
        import polars as pl
        return pl.DataFrame({
            "text": ["sample1", "sample2", "sample3"],
            "tokens": [100, 200, 300],
            "loss": [0.1, 0.2, 0.3],
        })
    
    def read_csv(self, path, **kwargs):
        """Mock read_csv method."""
        return self.read_parquet(path, **kwargs)
    
    @property
    def read_count(self) -> int:
        """Get number of reads."""
        return self._read_count


class TestDataGenerator:
    """Generator for test data."""
    
    @staticmethod
    def generate_prompts(num: int = 10) -> List[str]:
        """Generate test prompts."""
        return [f"Test prompt {i}" for i in range(num)]
    
    @staticmethod
    def generate_text_data(num: int = 100) -> Dict[str, List]:
        """Generate test text data."""
        return {
            "text": [f"Sample text {i}" for i in range(num)],
            "tokens": list(range(100, 100 + num)),
            "loss": [0.1 * (i % 10) for i in range(num)],
        }
    
    @staticmethod
    def generate_config() -> Dict[str, Any]:
        """Generate test configuration."""
        return {
            "model": {
                "name": "test-model",
                "path": "/tmp/test-model",
            },
            "inference": {
                "max_tokens": 64,
                "temperature": 0.7,
            },
        }













