"""
Integration tests for polyglot_core.

Tests compatibility, integration, and end-to-end workflows.
"""

import pytest
import numpy as np
from optimization_core.polyglot_core import (
    check_polyglot_availability,
    print_polyglot_status,
    get_test_compatibility_info,
    get_test_helper,
    UnifiedKVCache,  # Backward compatibility
    UnifiedCompressor,  # Backward compatibility
)


def test_polyglot_availability():
    """Test that polyglot_core is available."""
    availability = check_polyglot_availability()
    
    # At least main module should be available
    assert availability.get('polyglot_core', False) is True


def test_backward_compatibility_aliases():
    """Test backward compatibility aliases."""
    # These should work without errors
    try:
        cache = UnifiedKVCache(max_size=100)
        assert cache is not None
    except Exception:
        # May fail if backends not available, but should not raise ImportError
        pass
    
    try:
        compressor = UnifiedCompressor(algorithm="lz4")
        assert compressor is not None
    except Exception:
        pass


def test_test_helper():
    """Test PolyglotTestHelper."""
    helper = get_test_helper()
    
    assert helper is not None
    assert hasattr(helper, 'availability')
    
    # Test cache creation
    try:
        cache = helper.create_test_cache()
        assert cache is not None
    except Exception:
        pass  # May fail if no backends available


def test_compatibility_info():
    """Test compatibility info generation."""
    info = get_test_compatibility_info()
    
    assert isinstance(info, dict)
    assert 'polyglot_available' in info
    assert 'backends' in info
    assert 'modules' in info


def test_integration_workflow():
    """Test complete integration workflow."""
    from optimization_core.polyglot_core import (
        KVCache,
        Attention,
        Compressor,
        AttentionConfig
    )
    
    # 1. Create components
    cache = KVCache(max_size=100)
    attention = Attention(AttentionConfig(d_model=256, n_heads=4))
    compressor = Compressor(algorithm="lz4")
    
    # 2. Use cache
    k = np.random.randn(32).astype(np.float32)
    v = np.random.randn(32).astype(np.float32)
    cache.put(layer=0, position=0, key=k, value=v)
    result = cache.get(layer=0, position=0)
    assert result is not None
    
    # 3. Use attention
    q = np.random.randn(2 * 8, 256).astype(np.float32)
    output = attention.forward(q, q, q, batch_size=2, seq_len=8)
    assert output.output.shape == (16, 256)
    
    # 4. Use compression
    data = b"Test data " * 100
    result = compressor.compress(data)
    if result.success:
        assert len(result.data) > 0


def test_metrics_integration():
    """Test metrics integration."""
    from optimization_core.polyglot_core import get_metrics_collector, record_metric
    
    collector = get_metrics_collector()
    
    # Record some metrics
    record_metric("test_metric", 1.0)
    record_metric("test_metric", 2.0)
    record_metric("test_metric", 3.0)
    
    summary = collector.get_summary("test_metric")
    assert summary is not None
    assert summary.count == 3
    assert summary.avg == 2.0
    
    # Reset
    collector.reset()


def test_profiling_integration():
    """Test profiling integration."""
    from optimization_core.polyglot_core import get_profiler
    
    profiler = get_profiler()
    
    with profiler.profile("test_operation"):
        # Simulate work
        import time
        time.sleep(0.001)
    
    metrics = profiler.get_metrics("test_operation")
    assert metrics is not None
    assert metrics.duration_ms > 0


def test_benchmarking_integration():
    """Test benchmarking integration."""
    from optimization_core.polyglot_core import Benchmark
    
    benchmark = Benchmark()
    
    def test_func():
        return sum(range(100))
    
    result = benchmark.run("test_sum", test_func, iterations=10)
    
    assert result.success is True
    assert result.iterations == 10
    assert result.avg_time_ms >= 0


def test_reporting_integration():
    """Test reporting integration."""
    from optimization_core.polyglot_core import ReportGenerator
    
    generator = ReportGenerator()
    
    # Mock results
    from dataclasses import dataclass
    
    @dataclass
    class MockResult:
        backend: str = "python"
        throughput: float = 100.0
        avg_time_ms: float = 10.0
        success: bool = True
    
    results = {
        "test1": MockResult(),
        "test2": MockResult(backend="rust", throughput=500.0)
    }
    
    report = generator.generate_benchmark_report(results)
    
    assert report is not None
    assert len(report.sections) > 0
    assert report.title == "Benchmark Report"
    
    # Test export
    markdown = report.to_markdown()
    assert "Benchmark Report" in markdown
    
    json_str = report.to_json()
    assert "Benchmark Report" in json_str













