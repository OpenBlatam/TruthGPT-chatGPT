"""
Benchmark Utilities

Utility functions for benchmark analysis and reporting.
"""

from typing import Dict, List, Any
import numpy as np
import logging
from .benchmark_models import BenchmarkResult, ClosedSourceResult, BenchmarkReport

logger = logging.getLogger(__name__)


def generate_summary(
    polyglot_results: List[BenchmarkResult],
    closed_source_results: List[ClosedSourceResult]
) -> Dict[str, Any]:
    """Generar resumen comparativo."""
    summary = {
        'total_polyglot_tests': len(polyglot_results),
        'successful_polyglot_tests': sum(1 for r in polyglot_results if r.success),
        'total_closed_source_tests': len(closed_source_results),
        'successful_closed_source_tests': sum(1 for r in closed_source_results if r.success),
        'module_performance': {},
        'comparison': {}
    }
    
    # Agrupar por módulo
    by_module = {}
    for result in polyglot_results:
        if result.success:
            if result.module not in by_module:
                by_module[result.module] = []
            by_module[result.module].append(result)
    
    for module, results in by_module.items():
        if results:
            avg_latency = np.mean([r.latency_ms for r in results])
            avg_throughput = np.mean([r.throughput_tokens_per_sec for r in results])
            summary['module_performance'][module] = {
                'avg_latency_ms': avg_latency,
                'avg_throughput_tokens_per_sec': avg_throughput,
                'test_count': len(results)
            }
    
    # Comparar con closed source (solo inference)
    polyglot_inference = [r for r in polyglot_results if r.module == 'inference' and r.success]
    closed_source_inference = [r for r in closed_source_results if r.success]
    
    if polyglot_inference and closed_source_inference:
        polyglot_avg_latency = np.mean([r.latency_ms for r in polyglot_inference])
        polyglot_avg_throughput = np.mean([r.throughput_tokens_per_sec for r in polyglot_inference])
        
        closed_source_avg_latency = np.mean([r.latency_ms for r in closed_source_inference])
        closed_source_avg_throughput = np.mean([r.throughput_tokens_per_sec for r in closed_source_inference])
        
        summary['comparison'] = {
            'latency_speedup': closed_source_avg_latency / polyglot_avg_latency if polyglot_avg_latency > 0 else 0,
            'throughput_speedup': polyglot_avg_throughput / closed_source_avg_throughput if closed_source_avg_throughput > 0 else 0,
            'polyglot_avg_latency_ms': polyglot_avg_latency,
            'closed_source_avg_latency_ms': closed_source_avg_latency,
            'polyglot_avg_throughput_tokens_per_sec': polyglot_avg_throughput,
            'closed_source_avg_throughput_tokens_per_sec': closed_source_avg_throughput
        }
    
    return summary


def print_report(report: BenchmarkReport):
    """Imprimir reporte en consola."""
    print("\n" + "=" * 80)
    print("REPORTE DE BENCHMARKS - POLYGLOT vs CLOSED SOURCE")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp}")
    print(f"\nEstado de Módulos:")
    for module, status in report.module_status.items():
        status_str = "✓ Disponible" if status else "✗ No disponible"
        print(f"  {module}: {status_str}")
    
    print(f"\nResultados Polyglot ({len(report.polyglot_results)} tests):")
    successful = [r for r in report.polyglot_results if r.success]
    print(f"  Exitosos: {len(successful)}/{len(report.polyglot_results)}")
    
    by_module = {}
    for result in successful:
        if result.module not in by_module:
            by_module[result.module] = []
        by_module[result.module].append(result)
    
    for module, results in by_module.items():
        print(f"\n  {module.upper()}:")
        for result in results:
            print(f"    {result.name} ({result.backend}): "
                  f"{result.latency_ms:.2f}ms, "
                  f"{result.throughput_tokens_per_sec:.1f} tokens/s")
    
    print(f"\nResultados Closed Source ({len(report.closed_source_results)} tests):")
    successful_cs = [r for r in report.closed_source_results if r.success]
    print(f"  Exitosos: {len(successful_cs)}/{len(report.closed_source_results)}")
    for result in successful_cs:
        print(f"    {result.model_name}: "
              f"{result.latency_ms:.2f}ms, "
              f"{result.throughput_tokens_per_sec:.1f} tokens/s")
    
    if report.summary.get('comparison'):
        comp = report.summary['comparison']
        print(f"\nComparación:")
        if comp.get('latency_speedup', 0) > 0:
            print(f"  Latency: Polyglot es {comp['latency_speedup']:.2f}x {'más rápido' if comp['latency_speedup'] > 1 else 'más lento'}")
        if comp.get('throughput_speedup', 0) > 0:
            print(f"  Throughput: Polyglot es {comp['throughput_speedup']:.2f}x {'más rápido' if comp['throughput_speedup'] > 1 else 'más lento'}")
    
    print("=" * 80 + "\n")













