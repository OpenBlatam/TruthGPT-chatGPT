"""
Enhanced MCTS Optimizer with Neural Guidance and Olympiad Benchmarking.
Combines advanced MCTS techniques with mathematical reasoning evaluation.
"""

import random
import math
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .mcts_optimization import (
    MCTSOptimizationArgs, NeuralGuidedMCTSArgs, EnhancedMCTSOptimizer,
    create_enhanced_mcts_optimizer
)
from .olympiad_benchmarks import (
    OlympiadBenchmarkSuite, OlympiadBenchmarkConfig,
    get_olympiad_benchmark_config, create_olympiad_benchmark_suite
)

@dataclass
class EnhancedMCTSBenchmarkArgs:
    """Arguments for enhanced MCTS with olympiad benchmarking."""
    mcts_args: NeuralGuidedMCTSArgs = field(default_factory=NeuralGuidedMCTSArgs)
    benchmark_config: OlympiadBenchmarkConfig = field(default_factory=OlympiadBenchmarkConfig)
    enable_mathematical_reasoning: bool = True
    reasoning_weight: float = 0.2
    performance_weight: float = 0.8
    benchmark_frequency: int = 10

class SimplePolicyNetwork(nn.Module):
    """Simple policy network for MCTS guidance."""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, output_dim: int = 16):
        super().__init__()
        try:
            from optimization_core.cuda_kernels import OptimizedLinear
            linear_layer = OptimizedLinear
        except ImportError:
            linear_layer = nn.Linear
        
        self.network = nn.Sequential(
            linear_layer(input_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleValueNetwork(nn.Module):
    """Simple value network for MCTS guidance."""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        try:
            from optimization_core.cuda_kernels import OptimizedLinear
            linear_layer = OptimizedLinear
        except ImportError:
            linear_layer = nn.Linear
        
        self.network = nn.Sequential(
            linear_layer(input_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class EnhancedMCTSWithBenchmarks:
    """Enhanced MCTS optimizer with integrated olympiad benchmarking."""
    
    def __init__(self, args: EnhancedMCTSBenchmarkArgs, objective_function: Callable):
        self.args = args
        self.objective_function = objective_function
        
        self.policy_network = SimplePolicyNetwork()
        self.value_network = SimpleValueNetwork()
        
        self.mcts_optimizer = create_enhanced_mcts_optimizer(
            objective_function=self._combined_objective,
            args=args.mcts_args,
            policy_network=self.policy_network,
            value_network=self.value_network
        )
        
        self.benchmark_suite = OlympiadBenchmarkSuite(args.benchmark_config)
        self.benchmark_problems = []
        self.benchmark_results = []
        self.optimization_history = []
        
        if args.enable_mathematical_reasoning:
            self.benchmark_problems = self.benchmark_suite.generate_problem_set()
            print(f"Generated {len(self.benchmark_problems)} olympiad problems for benchmarking")
    
    def _combined_objective(self, config: Dict[str, Any]) -> float:
        """Combined objective function including performance and mathematical reasoning."""
        performance_score = self.objective_function(config)
        
        if not self.args.enable_mathematical_reasoning:
            return float(performance_score)
        
        reasoning_score = self._evaluate_mathematical_reasoning(config)
        
        performance_weight = float(self.args.performance_weight)
        reasoning_weight = float(self.args.reasoning_weight)
        performance_score = float(performance_score)
        reasoning_score = float(reasoning_score)
        
        combined_score = performance_weight * performance_score + reasoning_weight * reasoning_score
        
        return float(combined_score)
    
    def _evaluate_mathematical_reasoning(self, config: Dict[str, Any]) -> float:
        """Evaluate mathematical reasoning capabilities using olympiad problems."""
        if not self.benchmark_problems:
            return 0.5
        
        sample_size = min(5, len(self.benchmark_problems))
        sample_problems = random.sample(self.benchmark_problems, sample_size)
        
        correct_answers = 0
        for problem in sample_problems:
            try:
                answer = self._solve_problem_with_config(problem, config)
                if self._check_answer(problem, answer):
                    correct_answers += 1
            except Exception:
                pass
        
        accuracy = correct_answers / sample_size
        reasoning_score = max(0.0, 1.0 - accuracy)
        
        return reasoning_score
    
    def _solve_problem_with_config(self, problem, config: Dict[str, Any]) -> str:
        """Mock problem solving with given configuration."""
        difficulty_scores = {
            'amc_12': 0.8,
            'aime': 0.6,
            'usamo': 0.4,
            'imo': 0.2
        }
        
        base_score = difficulty_scores.get(problem.difficulty.value, 0.5)
        
        config_bonus = 0
        if config.get('learning_rate', 0) < 1e-3:
            config_bonus += 0.1
        if config.get('num_layers', 0) >= 6:
            config_bonus += 0.1
        if config.get('hidden_size', 0) >= 512:
            config_bonus += 0.1
        
        success_prob = min(0.95, base_score + config_bonus)
        
        if random.random() < success_prob:
            return str(problem.answer)
        else:
            return "incorrect_answer"
    
    def _check_answer(self, problem, answer: str) -> bool:
        """Check if answer is correct."""
        try:
            expected = str(problem.answer).lower().strip()
            given = answer.lower().strip()
            return expected == given
        except:
            return False
    
    def optimize_with_benchmarks(self, initial_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Run optimization with integrated benchmarking."""
        print("Starting Enhanced MCTS Optimization with Olympiad Benchmarking")
        print("=" * 70)
        
        start_time = time.time()
        
        best_config, best_score = self.mcts_optimizer.optimize(initial_config)
        
        optimization_time = time.time() - start_time
        
        final_benchmark_results = self._run_comprehensive_benchmark(best_config)
        
        optimization_stats = {
            'optimization_time': optimization_time,
            'best_score': best_score,
            'best_config': best_config,
            'benchmark_results': final_benchmark_results,
            'mcts_stats': getattr(self.mcts_optimizer, 'optimization_history', [])
        }
        
        self.optimization_history.append(optimization_stats)
        
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Best score: {best_score:.4f}")
        print(f"Mathematical reasoning accuracy: {final_benchmark_results.get('overall_accuracy', 0):.2%}")
        
        return best_config, best_score, optimization_stats
    
    def _run_comprehensive_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive olympiad benchmark evaluation."""
        if not self.benchmark_problems:
            return {'overall_accuracy': 0.0, 'category_performance': {}}
        
        print(f"\nRunning comprehensive benchmark with {len(self.benchmark_problems)} problems...")
        
        model_responses = []
        for i, problem in enumerate(self.benchmark_problems):
            if i % 10 == 0:
                print(f"Solving problem {i+1}/{len(self.benchmark_problems)}")
            
            response = self._solve_problem_with_config(problem, config)
            model_responses.append(response)
        
        results = self.benchmark_suite.evaluate_model_performance(model_responses)
        
        print(f"Benchmark completed. Overall accuracy: {results['overall_accuracy']:.2%}")
        
        return results
    
    def get_optimization_report(self) -> str:
        """Generate comprehensive optimization and benchmarking report."""
        if not self.optimization_history:
            return "No optimization runs completed yet."
        
        latest_run = self.optimization_history[-1]
        
        report = ["# Enhanced MCTS Optimization Report", ""]
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Optimization Time:** {latest_run['optimization_time']:.2f} seconds")
        report.append(f"**Best Score:** {latest_run['best_score']:.4f}")
        report.append("")
        
        report.append("## Best Configuration")
        for key, value in latest_run['best_config'].items():
            report.append(f"- **{key}**: {value}")
        report.append("")
        
        benchmark_results = latest_run['benchmark_results']
        if benchmark_results:
            report.append("## Mathematical Reasoning Performance")
            report.append(f"**Overall Accuracy:** {benchmark_results['overall_accuracy']:.2%}")
            report.append("")
            
            if 'category_performance' in benchmark_results:
                report.append("### Performance by Category")
                for category, perf in benchmark_results['category_performance'].items():
                    accuracy = perf['accuracy']
                    total = perf['total']
                    report.append(f"- **{category.replace('_', ' ').title()}**: {accuracy:.2%} ({perf['correct']}/{total})")
                report.append("")
            
            if 'difficulty_performance' in benchmark_results:
                report.append("### Performance by Difficulty")
                for difficulty, perf in benchmark_results['difficulty_performance'].items():
                    accuracy = perf['accuracy']
                    total = perf['total']
                    report.append(f"- **{difficulty.replace('_', ' ').upper()}**: {accuracy:.2%} ({perf['correct']}/{total})")
        
        report.append("")
        report.append("## MCTS Optimization Statistics")
        mcts_stats = latest_run.get('mcts_stats', [])
        if mcts_stats:
            latest_mcts = mcts_stats[-1]
            if 'stats' in latest_mcts:
                stats = latest_mcts['stats']
                report.append(f"- **Pruning Rate**: {stats.get('pruning_rate', 0):.2%}")
                report.append(f"- **Neural Guidance**: {'Enabled' if stats.get('neural_guidance_enabled', False) else 'Disabled'}")
                report.append(f"- **Total Evaluations**: {stats.get('total_evaluations', 0)}")
        
        return "\n".join(report)

def create_enhanced_mcts_with_benchmarks(
    objective_function: Callable,
    variant_name: str = 'default',
    custom_args: Optional[EnhancedMCTSBenchmarkArgs] = None
) -> EnhancedMCTSWithBenchmarks:
    """Factory function to create enhanced MCTS with olympiad benchmarks."""
    
    if custom_args is None:
        mcts_args = NeuralGuidedMCTSArgs(
            fe_max=100,
            init_size=10,
            use_neural_guidance=True,
            entropy_weight=0.1,
            pruning_threshold=0.01
        )
        
        benchmark_config = get_olympiad_benchmark_config(variant_name)
        
        args = EnhancedMCTSBenchmarkArgs(
            mcts_args=mcts_args,
            benchmark_config=benchmark_config
        )
    else:
        args = custom_args
    
    return EnhancedMCTSWithBenchmarks(args, objective_function)

def benchmark_mcts_comparison(variant_name: str = 'deepseek_v3', num_runs: int = 3):
    """Benchmark comparison between original and enhanced MCTS."""
    print("MCTS Performance Comparison Benchmark")
    print("=" * 50)
    
    def mock_objective(config):
        penalty = 0
        if config.get('learning_rate', 1e-4) > 1e-3:
            penalty += 0.1
        if config.get('dropout', 0.1) > 0.2:
            penalty += 0.05
        if config.get('num_layers', 6) > 8:
            penalty += 0.1
        
        base_loss = 0.5 + penalty + random.gauss(0, 0.05)
        return max(0.1, base_loss)
    
    print(f"Testing {num_runs} runs for variant: {variant_name}")
    
    original_times = []
    enhanced_times = []
    original_scores = []
    enhanced_scores = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        from .mcts_optimization import create_mcts_optimizer, MCTSOptimizationArgs
        
        original_args = MCTSOptimizationArgs(fe_max=50, init_size=5)
        original_optimizer = create_mcts_optimizer(mock_objective, original_args)
        
        start_time = time.time()
        orig_config, orig_score = original_optimizer.optimize()
        orig_time = time.time() - start_time
        
        original_times.append(orig_time)
        original_scores.append(orig_score)
        
        enhanced_optimizer = create_enhanced_mcts_with_benchmarks(
            mock_objective, variant_name
        )
        enhanced_optimizer.args.mcts_args.fe_max = 50
        enhanced_optimizer.args.mcts_args.init_size = 5
        enhanced_optimizer.args.benchmark_config.problems_per_category = 3
        
        start_time = time.time()
        enh_config, enh_score, enh_stats = enhanced_optimizer.optimize_with_benchmarks()
        enh_time = time.time() - start_time
        
        enhanced_times.append(enh_time)
        enhanced_scores.append(enh_score)
        
        print(f"  Original: {orig_score:.4f} in {orig_time:.2f}s")
        print(f"  Enhanced: {enh_score:.4f} in {enh_time:.2f}s")
    
    print(f"\n{'='*50}")
    print("BENCHMARK RESULTS")
    print(f"{'='*50}")
    
    avg_orig_time = sum(original_times) / len(original_times)
    avg_enh_time = sum(enhanced_times) / len(enhanced_times)
    avg_orig_score = sum(original_scores) / len(original_scores)
    avg_enh_score = sum(enhanced_scores) / len(enhanced_scores)
    
    print(f"Original MCTS:")
    print(f"  Average time: {avg_orig_time:.2f}s")
    print(f"  Average score: {avg_orig_score:.4f}")
    
    print(f"\nEnhanced MCTS:")
    print(f"  Average time: {avg_enh_time:.2f}s")
    print(f"  Average score: {avg_enh_score:.4f}")
    
    time_improvement = (avg_orig_time - avg_enh_time) / avg_orig_time * 100
    score_improvement = (avg_orig_score - avg_enh_score) / avg_orig_score * 100
    
    print(f"\nImprovements:")
    print(f"  Time: {time_improvement:+.1f}%")
    print(f"  Score: {score_improvement:+.1f}%")
    
    return {
        'original_times': original_times,
        'enhanced_times': enhanced_times,
        'original_scores': original_scores,
        'enhanced_scores': enhanced_scores,
        'time_improvement': time_improvement,
        'score_improvement': score_improvement
    }

if __name__ == "__main__":
    benchmark_mcts_comparison()
