"""
Run comparative benchmark for best open source and closed source models
"""

import asyncio
import sys
import os

sys.path.append('/home/ubuntu/TruthGPT')

from benchmarking_framework.comparative_benchmark import ComparativeBenchmark
from benchmarking_framework.model_registry import ModelRegistry

async def main():
    """Run comprehensive benchmark for best models only."""
    print("🚀 TruthGPT vs Best Open Source vs Best Closed Source Models")
    print("=" * 70)
    
    registry = ModelRegistry()
    best_models = registry.get_best_models_only()
    
    print("\n📋 Models to be benchmarked:")
    print("\n🤖 TruthGPT Models:")
    for model in best_models["truthgpt_models"]:
        if model:
            params = f"{model.parameters:,}" if model.parameters else "Unknown"
            print(f"  - {model.name} ({params} params)")
    
    print("\n🌟 Best Open Source Models:")
    for model in best_models["open_source_best"]:
        if model:
            params = f"{model.parameters:,}" if model.parameters else "Unknown"
            print(f"  - {model.name} ({params} params) - {model.provider}")
    
    print("\n🏢 Best Closed Source Models:")
    for model in best_models["closed_source_best"]:
        if model:
            params = f"{model.parameters:,}" if model.parameters else "Unknown"
            print(f"  - {model.name} ({params} params) - {model.provider}")
    
    print("\n🔄 Running comparative benchmark...")
    
    benchmark = ComparativeBenchmark()
    results = await benchmark.run_comparative_benchmark()
    
    print("\n📊 Benchmark Results Summary:")
    
    for category, models in results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for model in models:
            print(f"  - {model.model_name}")
            print(f"    Reasoning: {model.reasoning_accuracy:.1%}")
            print(f"    Math: {model.math_accuracy:.1%}")
            print(f"    Code: {model.code_accuracy:.1%}")
            print(f"    Inference: {model.inference_time_ms:.1f}ms")
            if model.cost_per_1k_tokens is not None:
                print(f"    Cost: ${model.cost_per_1k_tokens:.3f}/1K tokens")
            else:
                print(f"    Cost: Free (local)")
    
    print("\n📝 Generating comparative report...")
    report = benchmark.generate_comparative_report()
    
    report_path = "/home/ubuntu/TruthGPT/best_models_comparison_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    results_path = "/home/ubuntu/TruthGPT/best_models_comparison_results.json"
    benchmark.export_results(results_path)
    
    print(f"\n✅ Comparative benchmark completed!")
    print(f"📄 Report saved to: {report_path}")
    print(f"📊 Results saved to: {results_path}")
    
    print("\n🎯 Key Findings:")
    all_models = []
    for category in results.values():
        all_models.extend(category)
    
    if all_models:
        best_reasoning = max(all_models, key=lambda x: x.reasoning_accuracy)
        best_speed = min(all_models, key=lambda x: x.inference_time_ms)
        best_cost = min([m for m in all_models if m.cost_per_1k_tokens == 0.0], 
                       key=lambda x: x.reasoning_accuracy, default=None)
        
        print(f"- 🧮 Best Reasoning: {best_reasoning.model_name} ({best_reasoning.reasoning_accuracy:.1%})")
        print(f"- ⚡ Fastest: {best_speed.model_name} ({best_speed.inference_time_ms:.1f}ms)")
        if best_cost:
            print(f"- 💰 Best Value: {best_cost.model_name} (Free + {best_cost.reasoning_accuracy:.1%} accuracy)")

if __name__ == "__main__":
    asyncio.run(main())
