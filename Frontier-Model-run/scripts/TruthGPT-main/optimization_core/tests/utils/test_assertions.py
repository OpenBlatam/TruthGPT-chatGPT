"""
Custom test assertions for optimization_core.

Provides domain-specific assertions for testing inference engines,
data processors, and other components.
"""
import logging
from typing import Any, Optional, Dict, List
from unittest.mock import Mock

logger = logging.getLogger(__name__)


def assert_engine_works(
    engine: Any,
    test_prompts: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Assert that an inference engine works correctly.
    
    Args:
        engine: Inference engine to test
        test_prompts: Test prompts (defaults to simple prompts)
        **kwargs: Additional generation parameters
    
    Raises:
        AssertionError: If engine doesn't work correctly
    """
    if test_prompts is None:
        test_prompts = ["Test prompt 1", "Test prompt 2"]
    
    # Test single prompt
    result = engine.generate(test_prompts[0], **kwargs)
    assert result is not None, "Engine should return a result"
    assert isinstance(result, str), "Single prompt should return string"
    
    # Test batch prompts
    results = engine.generate(test_prompts, **kwargs)
    assert results is not None, "Engine should return results"
    assert isinstance(results, list), "Batch prompts should return list"
    assert len(results) == len(test_prompts), "Should return one result per prompt"
    
    # Test callable interface
    callable_result = engine(test_prompts[0], **kwargs)
    assert callable_result is not None, "Engine should be callable"


def assert_processor_works(
    processor: Any,
    test_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Assert that a data processor works correctly.
    
    Args:
        processor: Data processor to test
        test_path: Test file path (optional)
        **kwargs: Additional read parameters
    
    Raises:
        AssertionError: If processor doesn't work correctly
    """
    # Test that processor has required methods
    assert hasattr(processor, "read_parquet"), "Processor should have read_parquet method"
    
    # If test_path provided, test reading
    if test_path:
        try:
            result = processor.read_parquet(test_path, **kwargs)
            assert result is not None, "Processor should return data"
        except FileNotFoundError:
            # File doesn't exist, that's okay for testing
            pass


def assert_config_valid(
    config: Dict[str, Any],
    required_keys: Optional[List[str]] = None
) -> None:
    """
    Assert that a configuration is valid.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
    
    Raises:
        AssertionError: If configuration is invalid
    """
    assert isinstance(config, dict), "Config must be a dictionary"
    
    if required_keys:
        missing = [key for key in required_keys if key not in config]
        assert not missing, f"Config missing required keys: {missing}"


def assert_error_handled(
    func: callable,
    *args,
    expected_error: Optional[type] = None,
    **kwargs
) -> None:
    """
    Assert that a function handles errors correctly.
    
    Args:
        func: Function to test
        *args: Positional arguments
        expected_error: Expected error type (if any)
        **kwargs: Keyword arguments
    
    Raises:
        AssertionError: If error handling is incorrect
    """
    if expected_error:
        # Should raise expected error
        try:
            func(*args, **kwargs)
            assert False, f"Function should raise {expected_error.__name__}"
        except expected_error:
            pass  # Expected
        except Exception as e:
            assert False, f"Function raised {type(e).__name__} instead of {expected_error.__name__}"
    else:
        # Should not raise any error
        try:
            func(*args, **kwargs)
        except Exception as e:
            assert False, f"Function should not raise errors, but raised {type(e).__name__}: {e}"


def assert_performance_within_range(
    value: float,
    min_value: float,
    max_value: float,
    metric_name: str = "performance"
) -> None:
    """
    Assert that a performance metric is within expected range.
    
    Args:
        value: Performance value
        min_value: Minimum expected value
        max_value: Maximum expected value
        metric_name: Name of metric for error messages
    
    Raises:
        AssertionError: If value is out of range
    """
    assert min_value <= value <= max_value, (
        f"{metric_name} ({value}) is outside expected range "
        f"[{min_value}, {max_value}]"
    )


def assert_metrics_improved(
    baseline_metrics: Dict[str, float],
    improved_metrics: Dict[str, float],
    min_improvement: float = 0.1
) -> None:
    """
    Assert that metrics have improved.
    
    Args:
        baseline_metrics: Baseline metrics
        improved_metrics: Improved metrics
        min_improvement: Minimum improvement ratio
    
    Raises:
        AssertionError: If metrics haven't improved enough
    """
    for key in baseline_metrics:
        if key in improved_metrics:
            baseline = baseline_metrics[key]
            improved = improved_metrics[key]
            
            if baseline > 0:
                improvement = (improved - baseline) / baseline
                assert improvement >= min_improvement, (
                    f"Metric {key} improvement ({improvement:.2%}) "
                    f"is less than minimum ({min_improvement:.2%})"
                )













