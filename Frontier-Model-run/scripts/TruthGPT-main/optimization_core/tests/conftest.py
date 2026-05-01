"""
Pytest Configuration and Shared Fixtures

Centralized fixtures and configuration for all tests.
"""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND AVAILABILITY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def backend_availability():
    """Check which backends are available."""
    availability = {
        "rust": False,
        "cpp": False,
        "julia": False,
        "go": False,
        "python": True,
    }
    
    try:
        from truthgpt_rust import PyKVCache
        availability["rust"] = True
    except ImportError:
        pass
    
    try:
        import _cpp_core as cpp_core
        availability["cpp"] = True
    except ImportError:
        pass
    
    try:
        import julia
        from julia import TruthGPTCore
        availability["julia"] = True
    except ImportError:
        pass
    
    return availability

@pytest.fixture(scope="session")
def polyglot_modules():
    """Import and check polyglot modules."""
    modules = {}
    
    try:
        from optimization_core.polyglot import (
            KVCache, attention, Compressor, Tokenizer,
            get_available_backends
        )
        modules["polyglot"] = {
            "KVCache": KVCache,
            "attention": attention,
            "Compressor": Compressor,
            "Tokenizer": Tokenizer,
            "get_available_backends": get_available_backends,
        }
    except ImportError as e:
        logger.warning(f"Polyglot modules not available: {e}")
        modules["polyglot"] = None
    
    return modules

# ═══════════════════════════════════════════════════════════════════════════════
# TEST DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Hello, world!",
        "This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
    ]

@pytest.fixture
def sample_token_ids():
    """Sample token IDs for testing."""
    return [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500],
    ]

@pytest.fixture
def sample_tensors():
    """Sample tensors for testing."""
    try:
        import torch
        return {
            "q": torch.randn(2, 8, 128, 64),
            "k": torch.randn(2, 8, 128, 64),
            "v": torch.randn(2, 8, 128, 64),
        }
    except ImportError:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    try:
        from unittest.mock import MagicMock
        import torch
        
        model = MagicMock()
        model.parameters.return_value = [torch.randn(10, 10)]
        model.eval.return_value = model
        model.generate.return_value = torch.randint(0, 1000, (1, 10))
        return model
    except ImportError:
        return None

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    try:
        from unittest.mock import MagicMock
        
        tokenizer = MagicMock()
        tokenizer.encode.return_value = {"input_ids": [[1, 2, 3]]}
        tokenizer.decode.return_value = "Hello, world!"
        tokenizer.batch_decode.return_value = ["Hello, world!"]
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer
    except ImportError:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def benchmark_config():
    """Default benchmark configuration."""
    return {
        "num_runs": 10,
        "warmup_runs": 3,
        "timeout_seconds": 300,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MARKERS
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_rust: mark test as requiring Rust backend"
    )
    config.addinivalue_line(
        "markers", "requires_cpp: mark test as requiring C++ backend"
    )
    config.addinivalue_line(
        "markers", "requires_julia: mark test as requiring Julia backend"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark"
    )

