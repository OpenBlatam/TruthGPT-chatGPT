#!/usr/bin/env python3
"""
Test Comprehensivo de Top 10 Papers 2025
=========================================

Prueba cada paper individualmente, combinaciones y todos juntos.
Genera reporte detallado de métricas y cambios.
"""

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

# Import TruthGPT
from truthgpt_optimization_core_integration import (
    TruthGPTOptimizationCore,
    TruthGPTOptimizationCoreConfig
)


@dataclass
class TestResult:
    """Resultado de un test."""
    test_name: str
    papers_enabled: List[str]
    forward_time: float
    memory_usage_mb: float
    metrics: Dict[str, Any]
    output_shape: Tuple[int, ...]
    num_parameters: int
    error: str = None


class ComprehensiveTester:
    """Tester comprehensivo para los Top 10 Papers 2025."""
    
    def __init__(self, hidden_size: int = 768, vocab_size: int = 1000):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.results: List[TestResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
    
    def create_test_input(self, batch_size: int = 2, seq_len: int = 32) -> torch.Tensor:
        """Crea input de prueba."""
        return torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
    
    def count_parameters(self, model: TruthGPTOptimizationCore) -> int:
        """Cuenta parámetros del modelo."""
        return sum(p.numel() for p in model.model.parameters())
    
    def get_memory_usage(self) -> float:
        """Obtiene uso de memoria en MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def test_configuration(
        self,
        config: TruthGPTOptimizationCoreConfig,
        test_name: str,
        papers_enabled: List[str]
    ) -> TestResult:
        """Prueba una configuración específica."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {test_name}")
        print(f"📋 Papers enabled: {', '.join(papers_enabled) if papers_enabled else 'None'}")
        print(f"{'='*60}")
        
        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create model
            start_time = time.time()
            core = TruthGPTOptimizationCore(config)
            init_time = time.time() - start_time
            
            # Count parameters
            num_params = self.count_parameters(core)
            
            # Create test input
            input_ids = self.create_test_input(batch_size=2, seq_len=32)
            
            # Forward pass
            memory_before = self.get_memory_usage()
            forward_start = time.time()
            
            with torch.no_grad():
                output = core.model(input_ids)
            
            forward_time = time.time() - forward_start
            memory_after = self.get_memory_usage()
            memory_usage = memory_after - memory_before
            
            # Get metrics
            metrics = core.get_all_metrics()
            
            # Extract output shape
            output_shape = output['logits'].shape
            
            print(f"✅ Test completed successfully")
            print(f"   Forward time: {forward_time:.4f}s")
            print(f"   Memory usage: {memory_usage:.2f} MB")
            print(f"   Parameters: {num_params:,}")
            print(f"   Output shape: {output_shape}")
            
            result = TestResult(
                test_name=test_name,
                papers_enabled=papers_enabled,
                forward_time=forward_time,
                memory_usage_mb=memory_usage,
                metrics=metrics,
                output_shape=output_shape,
                num_parameters=num_params
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Error in test: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return TestResult(
                test_name=test_name,
                papers_enabled=papers_enabled,
                forward_time=0.0,
                memory_usage_mb=0.0,
                metrics={},
                output_shape=(0,),
                num_parameters=0,
                error=str(e)
            )
    
    def test_baseline(self) -> TestResult:
        """Test baseline sin papers."""
        config = TruthGPTOptimizationCoreConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            # Todos los papers deshabilitados
        )
        return self.test_configuration(config, "Baseline (No Papers)", [])
    
    def test_individual_papers(self) -> List[TestResult]:
        """Prueba cada paper individualmente."""
        papers = [
            ("Qwen3", {"enable_qwen3": True}),
            ("Absolute Zero", {"enable_absolute_zero": True}),
            ("Seed1.5-VL", {"enable_seed1_5_vl": True}),
            ("Mixture of Reasonings", {"enable_mixture_of_reasonings": True}),
            ("CRFT", {"enable_crft": True}),
            ("Meta-CoT", {"enable_meta_cot": True}),
            ("SFT vs RL", {"enable_sft_rl_generalization": True}),
            ("Learning Dynamics", {"enable_learning_dynamics": True}),
            ("Faster Cascades", {"enable_faster_cascades": True}),
            ("DeepSeek-V3", {"enable_deepseek_v3": True}),
        ]
        
        results = []
        for paper_name, paper_config in papers:
            config = TruthGPTOptimizationCoreConfig(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                **paper_config
            )
            result = self.test_configuration(
                config,
                f"Individual: {paper_name}",
                [paper_name]
            )
            results.append(result)
        
        return results
    
    def test_combinations(self) -> List[TestResult]:
        """Prueba combinaciones de papers."""
        combinations = [
            ("Reasoning Papers", {
                "enable_mixture_of_reasonings": True,
                "enable_meta_cot": True,
                "enable_crft": True
            }, ["Mixture of Reasonings", "Meta-CoT", "CRFT"]),
            
            ("RL Papers", {
                "enable_absolute_zero": True,
                "enable_sft_rl_generalization": True,
                "enable_meta_cot": True
            }, ["Absolute Zero", "SFT vs RL", "Meta-CoT"]),
            
            ("Multimodal Papers", {
                "enable_qwen3": True,
                "enable_seed1_5_vl": True
            }, ["Qwen3", "Seed1.5-VL"]),
            
            ("Efficiency Papers", {
                "enable_faster_cascades": True,
                "enable_deepseek_v3": True,
                "enable_crft": True
            }, ["Faster Cascades", "DeepSeek-V3", "CRFT"]),
            
            ("Training Papers", {
                "enable_learning_dynamics": True,
                "enable_sft_rl_generalization": True
            }, ["Learning Dynamics", "SFT vs RL"]),
        ]
        
        results = []
        for combo_name, combo_config, papers in combinations:
            config = TruthGPTOptimizationCoreConfig(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                **combo_config
            )
            result = self.test_configuration(
                config,
                f"Combination: {combo_name}",
                papers
            )
            results.append(result)
        
        return results
    
    def test_all_together(self) -> TestResult:
        """Prueba todos los papers juntos."""
        config = TruthGPTOptimizationCoreConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            enable_qwen3=True,
            enable_absolute_zero=True,
            enable_seed1_5_vl=True,
            enable_mixture_of_reasonings=True,
            enable_crft=True,
            enable_meta_cot=True,
            enable_sft_rl_generalization=True,
            enable_learning_dynamics=True,
            enable_faster_cascades=True,
            enable_deepseek_v3=True,
        )
        
        all_papers = [
            "Qwen3", "Absolute Zero", "Seed1.5-VL", "Mixture of Reasonings",
            "CRFT", "Meta-CoT", "SFT vs RL", "Learning Dynamics",
            "Faster Cascades", "DeepSeek-V3"
        ]
        
        return self.test_configuration(
            config,
            "All Papers Together",
            all_papers
        )
    
    def run_all_tests(self):
        """Ejecuta todos los tests."""
        print("\n" + "="*60)
        print("🚀 COMPREHENSIVE TESTING - TOP 10 PAPERS 2025")
        print("="*60)
        
        # Baseline
        print("\n📊 Step 1: Baseline Test")
        baseline = self.test_baseline()
        self.results.append(baseline)
        
        # Individual papers
        print("\n📊 Step 2: Individual Paper Tests")
        individual_results = self.test_individual_papers()
        self.results.extend(individual_results)
        
        # Combinations
        print("\n📊 Step 3: Combination Tests")
        combination_results = self.test_combinations()
        self.results.extend(combination_results)
        
        # All together
        print("\n📊 Step 4: All Papers Together")
        all_together = self.test_all_together()
        self.results.append(all_together)
        
        print("\n✅ All tests completed!")
    
    def generate_report(self) -> str:
        """Genera reporte detallado."""
        report = []
        report.append("="*80)
        report.append("📊 COMPREHENSIVE TEST REPORT - TOP 10 PAPERS 2025")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Baseline metrics
        baseline = next((r for r in self.results if r.test_name == "Baseline (No Papers)"), None)
        if baseline and not baseline.error:
            report.append("## 📈 BASELINE METRICS")
            report.append(f"- Forward Time: {baseline.forward_time:.4f}s")
            report.append(f"- Memory Usage: {baseline.memory_usage_mb:.2f} MB")
            report.append(f"- Parameters: {baseline.num_parameters:,}")
            report.append("")
        
        # Individual papers comparison
        report.append("## 🔬 INDIVIDUAL PAPERS ANALYSIS")
        report.append("")
        report.append("| Paper | Forward Time (s) | Memory (MB) | Parameters | Speedup | Memory Change |")
        report.append("|-------|------------------|-------------|------------|---------|---------------|")
        
        individual_results = [r for r in self.results if r.test_name.startswith("Individual:")]
        for result in individual_results:
            if result.error:
                report.append(f"| {result.test_name.replace('Individual: ', '')} | ERROR | - | - | - | - |")
                continue
            
            speedup = baseline.forward_time / result.forward_time if baseline and baseline.forward_time > 0 else 1.0
            memory_change = result.memory_usage_mb - baseline.memory_usage_mb if baseline else 0.0
            
            report.append(
                f"| {result.test_name.replace('Individual: ', '')} | "
                f"{result.forward_time:.4f} | "
                f"{result.memory_usage_mb:.2f} | "
                f"{result.num_parameters:,} | "
                f"{speedup:.2f}x | "
                f"{memory_change:+.2f} MB |"
            )
        
        report.append("")
        
        # Combinations analysis
        report.append("## 🔗 COMBINATIONS ANALYSIS")
        report.append("")
        report.append("| Combination | Papers | Forward Time (s) | Memory (MB) | Parameters |")
        report.append("|-------------|--------|------------------|-------------|------------|")
        
        combination_results = [r for r in self.results if r.test_name.startswith("Combination:")]
        for result in combination_results:
            if result.error:
                report.append(f"| {result.test_name.replace('Combination: ', '')} | ERROR | - | - | - |")
                continue
            
            papers_str = ", ".join(result.papers_enabled[:3])
            if len(result.papers_enabled) > 3:
                papers_str += f" +{len(result.papers_enabled)-3} more"
            
            report.append(
                f"| {result.test_name.replace('Combination: ', '')} | "
                f"{papers_str} | "
                f"{result.forward_time:.4f} | "
                f"{result.memory_usage_mb:.2f} | "
                f"{result.num_parameters:,} |"
            )
        
        report.append("")
        
        # All together
        all_together = next((r for r in self.results if r.test_name == "All Papers Together"), None)
        if all_together and not all_together.error:
            report.append("## 🌟 ALL PAPERS TOGETHER")
            report.append("")
            report.append(f"- Forward Time: {all_together.forward_time:.4f}s")
            report.append(f"- Memory Usage: {all_together.memory_usage_mb:.2f} MB")
            report.append(f"- Parameters: {all_together.num_parameters:,}")
            
            if baseline and not baseline.error:
                speedup = baseline.forward_time / all_together.forward_time
                memory_change = all_together.memory_usage_mb - baseline.memory_usage_mb
                param_change = all_together.num_parameters - baseline.num_parameters
                
                report.append("")
                report.append("### Comparison vs Baseline:")
                report.append(f"- Speedup: {speedup:.2f}x")
                report.append(f"- Memory Change: {memory_change:+.2f} MB")
                report.append(f"- Parameter Change: {param_change:+,}")
            report.append("")
        
        # Metrics summary
        report.append("## 📊 METRICS SUMMARY")
        report.append("")
        
        for result in self.results:
            if result.error or not result.metrics:
                continue
            
            report.append(f"### {result.test_name}")
            report.append("")
            
            # Extract key metrics
            for key, value in result.metrics.items():
                if isinstance(value, dict):
                    report.append(f"**{key}:**")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            report.append(f"  - {sub_key}: {sub_value:.4f}")
                        else:
                            report.append(f"  - {sub_key}: {sub_value}")
                elif isinstance(value, (int, float)):
                    report.append(f"- {key}: {value:.4f}")
            
            report.append("")
        
        # Key findings
        report.append("## 🔍 KEY FINDINGS")
        report.append("")
        
        if baseline and not baseline.error:
            # Find fastest
            valid_results = [r for r in self.results if not r.error and r != baseline]
            if valid_results:
                fastest = min(valid_results, key=lambda x: x.forward_time)
                slowest = max(valid_results, key=lambda x: x.forward_time)
                
                report.append(f"### ⚡ Performance")
                report.append(f"- Fastest: {fastest.test_name} ({fastest.forward_time:.4f}s)")
                report.append(f"- Slowest: {slowest.test_name} ({slowest.forward_time:.4f}s)")
                report.append("")
                
                # Memory analysis
                most_memory = max(valid_results, key=lambda x: x.memory_usage_mb)
                least_memory = min(valid_results, key=lambda x: x.memory_usage_mb)
                
                report.append(f"### 💾 Memory")
                report.append(f"- Most Memory: {most_memory.test_name} ({most_memory.memory_usage_mb:.2f} MB)")
                report.append(f"- Least Memory: {least_memory.test_name} ({least_memory.memory_usage_mb:.2f} MB)")
                report.append("")
        
        # Recommendations
        report.append("### 💡 Recommendations")
        report.append("")
        report.append("1. **For Speed**: Use Faster Cascades for inference optimization")
        report.append("2. **For Efficiency**: Use CRFT for parameter-efficient fine-tuning")
        report.append("3. **For Reasoning**: Combine Mixture of Reasonings + Meta-CoT")
        report.append("4. **For Multimodal**: Use Qwen3 or Seed1.5-VL")
        report.append("5. **For Training**: Use Learning Dynamics + SFT vs RL")
        report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, output_dir: Path = None):
        """Guarda resultados en JSON y reporte."""
        if output_dir is None:
            output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON
        json_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'test_name': r.test_name,
                    'papers_enabled': r.papers_enabled,
                    'forward_time': r.forward_time,
                    'memory_usage_mb': r.memory_usage_mb,
                    'num_parameters': r.num_parameters,
                    'output_shape': list(r.output_shape),
                    'error': r.error,
                    'metrics': r.metrics
                }
                for r in self.results
            ]
        }
        
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {json_file}")
        
        # Save report
        report = self.generate_report()
        report_file = output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_file}")
        
        # Print report to console
        print("\n" + report)


def main():
    """Función principal."""
    print("🚀 Starting Comprehensive Testing...")
    
    # Use 768 which is divisible by 12 (default num_attention_heads)
    tester = ComprehensiveTester(hidden_size=768, vocab_size=1000)
    tester.run_all_tests()
    tester.save_results()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()


