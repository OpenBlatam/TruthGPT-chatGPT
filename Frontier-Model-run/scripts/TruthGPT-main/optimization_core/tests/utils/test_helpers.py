"""
Test Helper Functions

Shared utilities for tests.
"""
import json
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_test_data_file(
    file_path: Union[str, Path],
    num_samples: int = 100,
    format: str = "jsonl"
) -> Path:
    """Create a test data file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            data = {
                "id": i,
                "text": f"This is sample text {i}",
                "label": i % 2,
            }
            
            if format == "jsonl":
                f.write(json.dumps(data) + "\n")
            elif format == "json":
                if i == 0:
                    f.write("[\n")
                f.write(json.dumps(data))
                if i < num_samples - 1:
                    f.write(",\n")
                else:
                    f.write("\n]")
    
    return file_path

def load_test_data(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load test data from file."""
    file_path = Path(file_path)
    
    if file_path.suffix == ".jsonl":
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    elif file_path.suffix == ".json":
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def assert_dict_contains(actual: Dict, expected: Dict, path: str = ""):
    """Assert that actual dict contains all keys from expected."""
    for key, expected_value in expected.items():
        full_path = f"{path}.{key}" if path else key
        
        assert key in actual, f"Missing key: {full_path}"
        
        if isinstance(expected_value, dict):
            assert isinstance(actual[key], dict), f"{full_path} is not a dict"
            assert_dict_contains(actual[key], expected_value, full_path)
        else:
            assert actual[key] == expected_value, (
                f"{full_path}: expected {expected_value}, got {actual[key]}"
            )

def assert_performance_improvement(
    baseline_ms: float,
    improved_ms: float,
    min_improvement: float = 1.1,
    tolerance: float = 0.05
):
    """Assert that performance improved by at least min_improvement."""
    if baseline_ms <= 0:
        raise ValueError("Baseline time must be positive")
    
    improvement = baseline_ms / improved_ms
    min_acceptable = min_improvement * (1 - tolerance)
    
    assert improvement >= min_acceptable, (
        f"Expected at least {min_improvement:.1f}x improvement, "
        f"got {improvement:.2f}x"
    )

def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to retry function on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

def skip_if_backend_unavailable(backend: str):
    """Decorator to skip test if backend unavailable."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if backend == "rust":
                    from truthgpt_rust import PyKVCache
                elif backend == "cpp":
                    import _cpp_core as cpp_core
                elif backend == "julia":
                    from julia import TruthGPTCore
                else:
                    raise ValueError(f"Unknown backend: {backend}")
                
                return func(*args, **kwargs)
            except ImportError:
                import pytest
                pytest.skip(f"Backend '{backend}' not available")
        return wrapper
    return decorator

def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed_ms:.2f}ms")
        return result, elapsed_ms
    return wrapper

__all__ = [
    "create_test_data_file",
    "load_test_data",
    "assert_dict_contains",
    "assert_performance_improvement",
    "retry_on_failure",
    "skip_if_backend_unavailable",
    "measure_time",
]

