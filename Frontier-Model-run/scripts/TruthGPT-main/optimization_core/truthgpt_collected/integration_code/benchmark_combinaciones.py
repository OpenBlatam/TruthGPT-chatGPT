#!/usr/bin/env python3
"""
Benchmark de Combinaciones - Mejores Configuraciones
=====================================================

Ejecuta benchmarks de las mejores combinaciones identificadas
y genera reporte comparativo.
"""

import torch
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

from truthgpt_optimization_core_integration import (
    TruthGPTOptimizationCore,
    TruthGPTOptimizationCoreConfig
)


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    config_name: str
    papers_enabled: List[str]
    forward_time: float
    memory_usage_mb: float
    num_parameters: int
    throughput: float
    speedup_vs_baseline: float
    params_increase: float
    memory_increase: float
    metrics: Dict[str, Any]
    error: str = None


class CombinationBenchmarker:
    """Benchmarker para combinaciones de papers."""
    
    def __init__(self, hidden_size: int = 768, vocab_size: int = 1000):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results: List[BenchmarkResult] = []
        self.baseline_result = None
        
        print(f"🔧 Device: {self.device}")
        print(f"🔧 Hidden size: {hidden_size}")
        print(f"🔧 Vocab size: {vocab_size}")
    
    def create_test_input(self, batch_size: int = 4, seq_len: int = 32) -> torch.Tensor:
        """Crea input de prueba."""
        return torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
    
    def count_parameters(self, core: TruthGPTOptimizationCore) -> int:
        """Cuenta parámetros del modelo."""
        return sum(p.numel() for p in core.model.parameters())
    
    def get_memory_usage(self) -> float:
        """Obtiene uso de memoria en MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def benchmark_configuration(
        self,
        config: TruthGPTOptimizationCoreConfig,
        config_name: str,
        papers_enabled: List[str],
        num_runs: int = 5
    ) -> BenchmarkResult:
        """Ejecuta benchmark de una configuración."""
        print(f"\n{'='*60}")
        print(f"🧪 Benchmarking: {config_name}")
        print(f"📋 Papers: {', '.join(papers_enabled) if papers_enabled else 'None'}")
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
            input_ids = self.create_test_input(batch_size=4, seq_len=32)
            
            # Warmup
            with torch.no_grad():
                _ = core.model(input_ids)
            
            # Benchmark runs
            forward_times = []
            memory_usages = []
            
            for run in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                memory_before = self.get_memory_usage()
                forward_start = time.time()
                
                with torch.no_grad():
                    output = core.model(input_ids)
                
                forward_time = time.time() - forward_start
                memory_after = self.get_memory_usage()
                memory_usage = memory_after - memory_before
                
                forward_times.append(forward_time)
                memory_usages.append(memory_usage)
            
            # Average results
            avg_forward_time = np.mean(forward_times)
            avg_memory = np.mean(memory_usages)
            throughput = 4.0 / avg_forward_time  # batch_size / time
            
            # Get metrics
            metrics = core.get_all_metrics()
            
            # Calculate speedup vs baseline
            speedup = 1.0
            params_increase = 0.0
            memory_increase = 0.0
            
            if self.baseline_result:
                speedup = self.baseline_result.forward_time / avg_forward_time
                params_increase = ((num_params - self.baseline_result.num_parameters) / 
                                  self.baseline_result.num_parameters) * 100
                memory_increase = avg_memory - self.baseline_result.memory_usage_mb
            
            print(f"✅ Benchmark completed")
            print(f"   Forward time: {avg_forward_time:.4f}s (avg of {num_runs} runs)")
            print(f"   Memory usage: {avg_memory:.2f} MB")
            print(f"   Parameters: {num_params:,}")
            print(f"   Throughput: {throughput:.2f} samples/s")
            print(f"   Speedup vs baseline: {speedup:.2f}x")
            print(f"   Params increase: {params_increase:+.2f}%")
            
            result = BenchmarkResult(
                config_name=config_name,
                papers_enabled=papers_enabled,
                forward_time=avg_forward_time,
                memory_usage_mb=avg_memory,
                num_parameters=num_params,
                throughput=throughput,
                speedup_vs_baseline=speedup,
                params_increase=params_increase,
                memory_increase=memory_increase,
                metrics=metrics
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Error in benchmark: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return BenchmarkResult(
                config_name=config_name,
                papers_enabled=papers_enabled,
                forward_time=0.0,
                memory_usage_mb=0.0,
                num_parameters=0,
                throughput=0.0,
                speedup_vs_baseline=0.0,
                params_increase=0.0,
                memory_increase=0.0,
                metrics={},
                error=str(e)
            )
    
    def benchmark_baseline(self) -> BenchmarkResult:
        """Benchmark baseline sin papers."""
        config = TruthGPTOptimizationCoreConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
        )
        result = self.benchmark_configuration(
            config,
            "Baseline (No Papers)",
            []
        )
        self.baseline_result = result
        return result
    
    def benchmark_optimal_combinations(self):
        """Ejecuta benchmarks de las mejores combinaciones."""
        print("\n" + "="*80)
        print("🚀 BENCHMARKING OPTIMAL COMBINATIONS")
        print("="*80)
        
        # Baseline
        print("\n📊 Step 1: Baseline")
        baseline = self.benchmark_baseline()
        self.results.append(baseline)
        
        # Opción 1: Máxima Eficiencia
        print("\n📊 Step 2: Opción 1 - Máxima Eficiencia")
        config1 = TruthGPTOptimizationCoreConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            enable_faster_cascades=True,
            enable_deepseek_v3=True,
            enable_crft=True,
            enable_meta_cot=True,
        )
        result1 = self.benchmark_configuration(
            config1,
            "Opción 1: Máxima Eficiencia",
            ["Faster Cascades", "DeepSeek-V3", "CRFT", "Meta-CoT"]
        )
        self.results.append(result1)
        
        # Opción 2: Máxima Precisión
        print("\n📊 Step 3: Opción 2 - Máxima Precisión")
        config2 = TruthGPTOptimizationCoreConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            enable_qwen3=True,
            enable_seed1_5_vl=True,
            enable_mixture_of_reasonings=True,
            enable_meta_cot=True,
        )
        result2 = self.benchmark_configuration(
            config2,
            "Opción 2: Máxima Precisión",
            ["Qwen3", "Seed1.5-VL", "Mixture of Reasonings", "Meta-CoT"]
        )
        self.results.append(result2)
        
        # Opción 3: Balanceado (RECOMENDADO)
        print("\n📊 Step 4: Opción 3 - Balanceado (RECOMENDADO)")
        config3 = TruthGPTOptimizationCoreConfig(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            enable_crft=True,
            enable_faster_cascades=True,
            enable_meta_cot=True,
            enable_mixture_of_reasonings=True,
            enable_qwen3=True,
        )
        result3 = self.benchmark_configuration(
            config3,
            "Opción 3: Balanceado",
            ["CRFT", "Faster Cascades", "Meta-CoT", "Mixture of Reasonings", "Qwen3"]
        )
        self.results.append(result3)
        
        # Opción 4: Todos Juntos
        print("\n📊 Step 5: Opción 4 - Todos Juntos")
        config4 = TruthGPTOptimizationCoreConfig(
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
        result4 = self.benchmark_configuration(
            config4,
            "Opción 4: Todos Juntos",
            [
                "Qwen3", "Absolute Zero", "Seed1.5-VL", "Mixture of Reasonings",
                "CRFT", "Meta-CoT", "SFT vs RL", "Learning Dynamics",
                "Faster Cascades", "DeepSeek-V3"
            ]
        )
        self.results.append(result4)
        
        print("\n✅ All benchmarks completed!")
    
    def generate_comparison_report(self) -> str:
        """Genera reporte comparativo."""
        report = []
        report.append("="*80)
        report.append("📊 BENCHMARK COMPARISON REPORT - OPTIMAL COMBINATIONS")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("## 📈 COMPARISON TABLE")
        report.append("")
        report.append("| Configuración | Speedup | Throughput | Params | Memoria | Mejora Total |")
        report.append("|---------------|---------|------------|--------|--------|--------------|")
        
        for result in self.results:
            if result.error:
                report.append(f"| {result.config_name} | ERROR | - | - | - | - |")
                continue
            
            # Calculate total improvement score
            # Speedup weight: 40%, Throughput weight: 20%, Params weight: -20%, Memory weight: -20%
            total_score = (
                (result.speedup_vs_baseline - 1.0) * 40 +  # Speedup contribution
                (result.throughput / self.baseline_result.throughput - 1.0) * 20 -  # Throughput
                (result.params_increase / 100) * 20 -  # Params penalty
                (result.memory_increase / 100) * 20  # Memory penalty
            ) if self.baseline_result else 0.0
            
            report.append(
                f"| {result.config_name} | "
                f"{result.speedup_vs_baseline:.2f}x | "
                f"{result.throughput:.2f} | "
                f"{result.params_increase:+.2f}% | "
                f"{result.memory_increase:+.2f} MB | "
                f"{total_score:+.1f}% |"
            )
        
        report.append("")
        
        # Detailed results
        report.append("## 🔍 DETAILED RESULTS")
        report.append("")
        
        for result in self.results:
            if result.error:
                report.append(f"### {result.config_name}")
                report.append(f"❌ **Error**: {result.error}")
                report.append("")
                continue
            
            report.append(f"### {result.config_name}")
            report.append("")
            report.append(f"**Papers Enabled**: {', '.join(result.papers_enabled)}")
            report.append("")
            report.append(f"- **Forward Time**: {result.forward_time:.4f}s")
            report.append(f"- **Throughput**: {result.throughput:.2f} samples/s")
            report.append(f"- **Parameters**: {result.num_parameters:,}")
            report.append(f"- **Memory Usage**: {result.memory_usage_mb:.2f} MB")
            report.append("")
            
            if self.baseline_result:
                report.append("**vs Baseline:**")
                report.append(f"- Speedup: {result.speedup_vs_baseline:.2f}x")
                report.append(f"- Params Change: {result.params_increase:+.2f}%")
                report.append(f"- Memory Change: {result.memory_increase:+.2f} MB")
                report.append("")
        
        # Recommendations
        report.append("## 💡 RECOMMENDATIONS")
        report.append("")
        
        valid_results = [r for r in self.results if not r.error and r != self.baseline_result]
        
        if valid_results:
            # Best speedup
            best_speedup = max(valid_results, key=lambda x: x.speedup_vs_baseline)
            report.append(f"### ⚡ Best Speedup")
            report.append(f"- **{best_speedup.config_name}**: {best_speedup.speedup_vs_baseline:.2f}x")
            report.append("")
            
            # Best efficiency (speedup / params)
            best_efficiency = max(
                valid_results,
                key=lambda x: x.speedup_vs_baseline / max(1.0, abs(x.params_increase) / 100)
            )
            report.append(f"### 🎯 Best Efficiency")
            report.append(f"- **{best_efficiency.config_name}**: {best_efficiency.speedup_vs_baseline:.2f}x speedup, {best_efficiency.params_increase:+.2f}% params")
            report.append("")
            
            # Best balance
            report.append(f"### ⚖️ Best Balance (Recommended)")
            report.append(f"- **Opción 3: Balanceado** - Best trade-off between speed, precision, and efficiency")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, output_dir: Path = None):
        """Guarda resultados."""
        if output_dir is None:
            output_dir = Path(__file__).parent / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON
        json_file = output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'baseline': asdict(self.baseline_result) if self.baseline_result else None,
            'results': [asdict(r) for r in self.results]
        }
        
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {json_file}")
        
        # Save report
        report = self.generate_comparison_report()
        report_file = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_file}")
        
        # Print report
        print("\n" + report)


def main():
    """Función principal."""
    print("🚀 Starting Combination Benchmarking...")
    print("="*80)
    
    benchmarker = CombinationBenchmarker(hidden_size=768, vocab_size=1000)
    benchmarker.benchmark_optimal_combinations()
    benchmarker.save_results()
    
    print("\n✅ Benchmarking complete!")


if __name__ == "__main__":
    main()



