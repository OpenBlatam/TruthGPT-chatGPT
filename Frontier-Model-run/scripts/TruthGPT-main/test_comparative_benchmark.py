"""
Test suite for comparative benchmarking framework
"""

import asyncio
import json
import sys
import os

sys.path.append('/home/ubuntu/TruthGPT')

from benchmarking_framework.comparative_benchmark import ComparativeBenchmark, ComparativeMetrics
from benchmarking_framework.model_registry import ModelRegistry

def test_model_registry_best_models():
    """Test that model registry returns only best models."""
    print("🧪 Testing model registry best models...")
    
    registry = ModelRegistry()
    best_models = registry.get_best_models_only()
    
    assert "truthgpt_models" in best_models
    assert "open_source_best" in best_models
    assert "closed_source_best" in best_models
    
    assert len(best_models["open_source_best"]) == 3
    assert len(best_models["closed_source_best"]) == 3
    assert len(best_models["truthgpt_models"]) == 4
    
    open_source_names = [m.name for m in best_models["open_source_best"] if m]
    expected_open_source = ["Llama-3.1-405B", "Qwen2.5-72B", "DeepSeek-V3"]
    
    for expected in expected_open_source:
        assert expected in open_source_names, f"Missing {expected} in open source best"
    
    closed_source_names = [m.name for m in best_models["closed_source_best"] if m]
    expected_closed_source = ["Claude-3.5-Sonnet", "GPT-4o", "Gemini-1.5-Pro"]
    
    for expected in expected_closed_source:
        assert expected in closed_source_names, f"Missing {expected} in closed source best"
    
    print("✅ Model registry best models test passed")

def test_comparative_metrics():
    """Test ComparativeMetrics dataclass."""
    print("🧪 Testing ComparativeMetrics...")
    
    metrics = ComparativeMetrics(
        model_name="Test Model",
        model_type="open_source",
        provider="Test Provider",
        parameters=1000000,
        context_length=2048,
        inference_time_ms=50.0,
        throughput_tokens_per_sec=1000.0,
        memory_usage_mb=500.0,
        reasoning_accuracy=0.85,
        math_accuracy=0.80,
        code_accuracy=0.75,
        multilingual_accuracy=0.70
    )
    
    assert metrics.model_name == "Test Model"
    assert metrics.parameters == 1000000
    assert metrics.reasoning_accuracy == 0.85
    assert metrics.cost_per_1k_tokens is None
    
    print("✅ ComparativeMetrics test passed")

async def test_comparative_benchmark():
    """Test comparative benchmark functionality."""
    print("🧪 Testing comparative benchmark...")
    
    benchmark = ComparativeBenchmark()
    
    registry = benchmark.registry
    best_models = registry.get_best_models_only()
    
    truthgpt_model = best_models["truthgpt_models"][0] if best_models["truthgpt_models"] else None
    
    if truthgpt_model:
        metrics = benchmark.benchmark_truthgpt_model(truthgpt_model)
        assert isinstance(metrics, ComparativeMetrics)
        assert metrics.model_name.startswith("TruthGPT")
        assert metrics.cost_per_1k_tokens == 0.0
        print(f"✅ TruthGPT model {metrics.model_name} benchmarked successfully")
    
    closed_source_model = best_models["closed_source_best"][0] if best_models["closed_source_best"] else None
    if closed_source_model:
        metrics = await benchmark.benchmark_closed_source_model(closed_source_model)
        assert isinstance(metrics, ComparativeMetrics)
        assert metrics.cost_per_1k_tokens is not None
        assert metrics.cost_per_1k_tokens > 0
        print(f"✅ Closed source model {metrics.model_name} benchmarked successfully")
    
    print("✅ Comparative benchmark test passed")

def test_report_generation():
    """Test report generation."""
    print("🧪 Testing report generation...")
    
    benchmark = ComparativeBenchmark()
    
    mock_results = {
        "truthgpt_models": [
            ComparativeMetrics(
                model_name="TruthGPT-DeepSeek-V3",
                model_type="truthgpt",
                provider="TruthGPT",
                parameters=1550312,
                context_length=2048,
                inference_time_ms=2.84,
                throughput_tokens_per_sec=1000.0,
                memory_usage_mb=2.80,
                reasoning_accuracy=0.85,
                math_accuracy=0.80,
                code_accuracy=0.75,
                multilingual_accuracy=0.70,
                cost_per_1k_tokens=0.0
            )
        ],
        "open_source_best": [
            ComparativeMetrics(
                model_name="Llama-3.1-405B",
                model_type="open_source",
                provider="Meta",
                parameters=405000000000,
                context_length=128000,
                inference_time_ms=50.0,
                throughput_tokens_per_sec=500.0,
                memory_usage_mb=1000.0,
                reasoning_accuracy=0.90,
                math_accuracy=0.85,
                code_accuracy=0.80,
                multilingual_accuracy=0.75,
                cost_per_1k_tokens=0.0
            )
        ],
        "closed_source_best": [
            ComparativeMetrics(
                model_name="Claude-3.5-Sonnet",
                model_type="closed_source",
                provider="Anthropic",
                parameters=None,
                context_length=200000,
                inference_time_ms=30.0,
                throughput_tokens_per_sec=800.0,
                memory_usage_mb=None,
                reasoning_accuracy=0.92,
                math_accuracy=0.88,
                code_accuracy=0.85,
                multilingual_accuracy=0.80,
                cost_per_1k_tokens=0.003
            )
        ]
    }
    
    benchmark.results = mock_results
    report = benchmark.generate_comparative_report()
    
    print(f"Generated report preview: {report[:200]}...")
    
    assert len(report) > 100, "Report should have substantial content"
    assert "TruthGPT" in report, "Report should mention TruthGPT"
    assert "Performance" in report or "Comparison" in report, "Report should have performance content"
    
    print("✅ Report generation test passed")

def test_export_functionality():
    """Test results export."""
    print("🧪 Testing export functionality...")
    
    benchmark = ComparativeBenchmark()
    
    mock_results = {
        "truthgpt_models": [
            ComparativeMetrics(
                model_name="TruthGPT-Test",
                model_type="truthgpt",
                provider="TruthGPT",
                parameters=1000000,
                context_length=2048,
                inference_time_ms=5.0,
                throughput_tokens_per_sec=1000.0,
                memory_usage_mb=10.0,
                reasoning_accuracy=0.85,
                math_accuracy=0.80,
                code_accuracy=0.75,
                multilingual_accuracy=0.70,
                cost_per_1k_tokens=0.0
            )
        ],
        "open_source_best": [],
        "closed_source_best": []
    }
    
    benchmark.results = mock_results
    export_path = "/tmp/test_comparative_results.json"
    benchmark.export_results(export_path)
    
    assert os.path.exists(export_path)
    
    with open(export_path, 'r') as f:
        exported_data = json.load(f)
    
    assert "truthgpt_models" in exported_data
    assert "open_source_best" in exported_data
    assert "closed_source_best" in exported_data
    
    os.remove(export_path)
    
    print("✅ Export functionality test passed")

async def main():
    """Run all tests."""
    print("🚀 Starting Comparative Benchmark Tests")
    print("=" * 50)
    
    test_model_registry_best_models()
    test_comparative_metrics()
    await test_comparative_benchmark()
    test_report_generation()
    test_export_functionality()
    
    print("\n🎉 All comparative benchmark tests passed!")
    print("✅ Framework ready for best models comparison")

if __name__ == "__main__":
    asyncio.run(main())
