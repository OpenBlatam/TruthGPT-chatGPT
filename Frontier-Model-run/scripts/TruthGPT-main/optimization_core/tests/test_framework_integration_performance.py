"""
Tests de rendimiento para medir el aumento de performance cuando se integra con el framework completo.

Este test suite mide:
1. Rendimiento de módulos individuales
2. Rendimiento cuando se combinan módulos
3. Mejora de rendimiento por integración
4. Overhead de integración
5. Escalabilidad con múltiples módulos

Usage:
    python tests/test_framework_integration_performance.py
    python tests/test_framework_integration_performance.py --detailed
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import numpy as np

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Resultado de una medición de rendimiento."""
    name: str
    module: str
    latency_ms: float
    throughput: float
    operations: int
    memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


@dataclass
class IntegrationComparison:
    """Comparación entre módulos individuales vs integrados."""
    individual_total_ms: float
    integrated_ms: float
    speedup: float
    overhead_ms: float
    overhead_percent: float
    efficiency: float  # integrated / individual (ideal = 1.0)
    
    def to_dict(self):
        return asdict(self)


class FrameworkPerformanceBenchmark:
    """Benchmark de rendimiento del framework integrado."""
    
    def __init__(self):
        self.results: List[PerformanceResult] = []
        self.comparisons: List[IntegrationComparison] = []
    
    def benchmark_individual_attention(self) -> PerformanceResult:
        """Benchmark de Attention individual."""
        try:
            # Try multiple import paths
            try:
                from polyglot_core.attention import Attention, AttentionConfig
            except ImportError:
                try:
                    from optimization_core.polyglot_core.attention import Attention, AttentionConfig
                except ImportError:
                    # Direct import from current directory
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from polyglot_core.attention import Attention, AttentionConfig
            
            config = AttentionConfig(d_model=768, n_heads=12)
            attention = Attention(config)
            
            batch_size = 4
            seq_len = 512
            d_model = 768
            
            q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            k = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            v = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            
            # Warmup
            for _ in range(3):
                _ = attention.forward(q, k, v, batch_size, seq_len)
            
            # Benchmark
            iterations = 10
            start = time.perf_counter()
            for _ in range(iterations):
                output = attention.forward(q, k, v, batch_size, seq_len)
            total_time_ms = (time.perf_counter() - start) * 1000
            
            avg_latency = total_time_ms / iterations
            throughput = (batch_size * seq_len * iterations) / (total_time_ms / 1000)
            
            result = PerformanceResult(
                name="attention_individual",
                module="attention",
                latency_ms=avg_latency,
                throughput=throughput,
                operations=iterations,
                metadata={
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'd_model': d_model
                }
            )
            
            logger.info(f"  Attention individual: {avg_latency:.2f}ms, {throughput:.0f} tokens/s")
            return result
            
        except Exception as e:
            logger.warning(f"Attention individual benchmark skipped: {e}")
            return PerformanceResult(
                name="attention_individual",
                module="attention",
                latency_ms=0.0,
                throughput=0.0,
                operations=0,
                metadata={'error': str(e)}
            )
    
    def benchmark_individual_compression(self) -> PerformanceResult:
        """Benchmark de Compression individual."""
        try:
            # Try multiple import paths
            try:
                from polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
            except ImportError:
                try:
                    from optimization_core.polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
                except ImportError:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
            
            compressor = Compressor(config=CompressionConfig(algorithm=CompressionAlgorithm.LZ4))
            test_data = b"test_data" * 1000
            
            # Warmup
            for _ in range(5):
                result = compressor.compress(test_data)
                _ = compressor.decompress(result.data)
            
            # Benchmark
            iterations = 100
            start = time.perf_counter()
            for _ in range(iterations):
                result = compressor.compress(test_data)
                _ = compressor.decompress(result.data)
            total_time_ms = (time.perf_counter() - start) * 1000
            
            avg_latency = total_time_ms / iterations
            throughput = (len(test_data) * iterations) / (total_time_ms / 1000) / 1024 / 1024  # MB/s
            
            result = PerformanceResult(
                name="compression_individual",
                module="compression",
                latency_ms=avg_latency,
                throughput=throughput,
                operations=iterations,
                metadata={'data_size_bytes': len(test_data)}
            )
            
            logger.info(f"  Compression individual: {avg_latency:.2f}ms, {throughput:.2f} MB/s")
            return result
            
        except Exception as e:
            logger.warning(f"Compression individual benchmark skipped: {e}")
            return PerformanceResult(
                name="compression_individual",
                module="compression",
                latency_ms=0.0,
                throughput=0.0,
                operations=0,
                metadata={'error': str(e)}
            )
    
    def benchmark_integrated_attention_cache(self) -> PerformanceResult:
        """Benchmark de Attention + Cache integrados."""
        try:
            # Try multiple import paths
            try:
                from polyglot_core.attention import Attention, AttentionConfig
                from polyglot_core.cache import KVCache, KVCacheConfig
            except ImportError:
                try:
                    from optimization_core.polyglot_core.attention import Attention, AttentionConfig
                    from optimization_core.polyglot_core.cache import KVCache, KVCacheConfig
                except ImportError:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from polyglot_core.attention import Attention, AttentionConfig
                    from polyglot_core.cache import KVCache, KVCacheConfig
            
            attention = Attention(AttentionConfig(d_model=768, n_heads=12))
            cache = KVCache(config=KVCacheConfig(max_size=1000))
            
            batch_size = 4
            seq_len = 512
            d_model = 768
            
            q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            k = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            v = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            
            # Warmup
            for _ in range(3):
                output = attention.forward(q, k, v, batch_size, seq_len)
                # Simular cache (guardar keys/values)
                cache.put(layer=0, position=0, key=k[:d_model], value=v[:d_model])
            
            # Benchmark integrado
            iterations = 10
            start = time.perf_counter()
            for i in range(iterations):
                # Ejecutar attention
                output = attention.forward(q, k, v, batch_size, seq_len)
                # Guardar en cache
                cache.put(layer=0, position=i % 10, key=k[:d_model], value=v[:d_model])
                # Recuperar de cache (simulado)
                cached = cache.get(layer=0, position=(i-1) % 10)
            total_time_ms = (time.perf_counter() - start) * 1000
            
            avg_latency = total_time_ms / iterations
            throughput = (batch_size * seq_len * iterations) / (total_time_ms / 1000)
            
            result = PerformanceResult(
                name="attention_cache_integrated",
                module="integration",
                latency_ms=avg_latency,
                throughput=throughput,
                operations=iterations,
                metadata={
                    'modules': ['attention', 'cache'],
                    'batch_size': batch_size,
                    'seq_len': seq_len
                }
            )
            
            logger.info(f"  Attention+Cache integrado: {avg_latency:.2f}ms, {throughput:.0f} tokens/s")
            return result
            
        except Exception as e:
            logger.warning(f"Attention+Cache integrated benchmark skipped: {e}")
            return PerformanceResult(
                name="attention_cache_integrated",
                module="integration",
                latency_ms=0.0,
                throughput=0.0,
                operations=0,
                metadata={'error': str(e)}
            )
    
    def benchmark_integrated_compression_cache(self) -> PerformanceResult:
        """Benchmark de Compression + Cache integrados."""
        try:
            # Try multiple import paths
            try:
                from polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
                from polyglot_core.cache import KVCache, KVCacheConfig
            except ImportError:
                try:
                    from optimization_core.polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
                    from optimization_core.polyglot_core.cache import KVCache, KVCacheConfig
                except ImportError:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
                    from polyglot_core.cache import KVCache, KVCacheConfig
            
            compressor = Compressor(config=CompressionConfig(algorithm=CompressionAlgorithm.LZ4))
            cache = KVCache(config=KVCacheConfig(max_size=1000))
            
            test_data = b"test_data" * 1000
            import numpy as np
            key = np.random.randn(64).astype(np.float32)
            value = np.random.randn(64).astype(np.float32)
            
            # Warmup
            for _ in range(5):
                compressed = compressor.compress(test_data)
                cache.put(layer=0, position=0, key=key, value=value)
            
            # Benchmark integrado
            iterations = 100
            start = time.perf_counter()
            for i in range(iterations):
                # Comprimir
                compressed = compressor.compress(test_data)
                # Guardar comprimido en cache (simulado)
                cache.put(layer=0, position=i % 10, key=key, value=value)
                # Recuperar y descomprimir
                cached = cache.get(layer=0, position=(i-1) % 10)
                if cached:
                    _ = compressor.decompress(compressed.data)
            total_time_ms = (time.perf_counter() - start) * 1000
            
            avg_latency = total_time_ms / iterations
            throughput = (len(test_data) * iterations) / (total_time_ms / 1000) / 1024 / 1024  # MB/s
            
            result = PerformanceResult(
                name="compression_cache_integrated",
                module="integration",
                latency_ms=avg_latency,
                throughput=throughput,
                operations=iterations,
                metadata={
                    'modules': ['compression', 'cache'],
                    'data_size_bytes': len(test_data)
                }
            )
            
            logger.info(f"  Compression+Cache integrado: {avg_latency:.2f}ms, {throughput:.2f} MB/s")
            return result
            
        except Exception as e:
            logger.warning(f"Compression+Cache integrated benchmark skipped: {e}")
            return PerformanceResult(
                name="compression_cache_integrated",
                module="integration",
                latency_ms=0.0,
                throughput=0.0,
                operations=0,
                metadata={'error': str(e)}
            )
    
    def benchmark_full_pipeline(self) -> PerformanceResult:
        """Benchmark del pipeline completo (Attention + Cache + Compression)."""
        try:
            # Try multiple import paths
            try:
                from polyglot_core.attention import Attention, AttentionConfig
                from polyglot_core.cache import KVCache, KVCacheConfig
                from polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
            except ImportError:
                try:
                    from optimization_core.polyglot_core.attention import Attention, AttentionConfig
                    from optimization_core.polyglot_core.cache import KVCache, KVCacheConfig
                    from optimization_core.polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
                except ImportError:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from polyglot_core.attention import Attention, AttentionConfig
                    from polyglot_core.cache import KVCache, KVCacheConfig
                    from polyglot_core.compression import Compressor, CompressionConfig, CompressionAlgorithm
            
            attention = Attention(AttentionConfig(d_model=768, n_heads=12))
            cache = KVCache(config=KVCacheConfig(max_size=1000))
            compressor = Compressor(config=CompressionConfig(algorithm=CompressionAlgorithm.LZ4))
            
            batch_size = 4
            seq_len = 512
            d_model = 768
            
            q = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            k = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            v = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
            
            # Warmup
            for _ in range(3):
                output = attention.forward(q, k, v, batch_size, seq_len)
                k_bytes = k[:d_model].tobytes()
                compressed = compressor.compress(k_bytes)
                cache.put(layer=0, position=0, key=k[:d_model], value=v[:d_model])
            
            # Benchmark pipeline completo
            iterations = 10
            start = time.perf_counter()
            for i in range(iterations):
                # 1. Attention
                output = attention.forward(q, k, v, batch_size, seq_len)
                # 2. Comprimir keys/values
                k_bytes = k[:d_model].tobytes()
                v_bytes = v[:d_model].tobytes()
                compressed_k = compressor.compress(k_bytes)
                compressed_v = compressor.compress(v_bytes)
                # 3. Guardar en cache
                cache.put(layer=0, position=i % 10, key=k[:d_model], value=v[:d_model])
                # 4. Recuperar de cache
                cached = cache.get(layer=0, position=(i-1) % 10)
                if cached:
                    # 5. Descomprimir (simulado)
                    _ = compressor.decompress(compressed_k.data)
            total_time_ms = (time.perf_counter() - start) * 1000
            
            avg_latency = total_time_ms / iterations
            throughput = (batch_size * seq_len * iterations) / (total_time_ms / 1000)
            
            result = PerformanceResult(
                name="full_pipeline",
                module="integration",
                latency_ms=avg_latency,
                throughput=throughput,
                operations=iterations,
                metadata={
                    'modules': ['attention', 'cache', 'compression'],
                    'batch_size': batch_size,
                    'seq_len': seq_len
                }
            )
            
            logger.info(f"  Pipeline completo: {avg_latency:.2f}ms, {throughput:.0f} tokens/s")
            return result
            
        except Exception as e:
            logger.warning(f"Full pipeline benchmark skipped: {e}")
            return PerformanceResult(
                name="full_pipeline",
                module="integration",
                latency_ms=0.0,
                throughput=0.0,
                operations=0,
                metadata={'error': str(e)}
            )
    
    def compare_individual_vs_integrated(self) -> List[IntegrationComparison]:
        """Comparar rendimiento individual vs integrado."""
        comparisons = []
        
        # Obtener resultados individuales
        attention_individual = next((r for r in self.results if r.name == "attention_individual"), None)
        compression_individual = next((r for r in self.results if r.name == "compression_individual"), None)
        attention_cache_integrated = next((r for r in self.results if r.name == "attention_cache_integrated"), None)
        compression_cache_integrated = next((r for r in self.results if r.name == "compression_cache_integrated"), None)
        full_pipeline = next((r for r in self.results if r.name == "full_pipeline"), None)
        
        # Comparación 1: Attention individual vs Attention+Cache
        if attention_individual and attention_cache_integrated:
            if attention_individual.latency_ms > 0 and attention_cache_integrated.latency_ms > 0:
                individual_total = attention_individual.latency_ms
                integrated = attention_cache_integrated.latency_ms
                overhead = integrated - individual_total
                overhead_percent = (overhead / individual_total) * 100 if individual_total > 0 else 0
                speedup = individual_total / integrated if integrated > 0 else 0
                efficiency = individual_total / integrated if integrated > 0 else 0
                
                comparison = IntegrationComparison(
                    individual_total_ms=individual_total,
                    integrated_ms=integrated,
                    speedup=speedup,
                    overhead_ms=overhead,
                    overhead_percent=overhead_percent,
                    efficiency=efficiency
                )
                comparisons.append(comparison)
                
                logger.info(f"\n  Comparación Attention vs Attention+Cache:")
                logger.info(f"    Individual: {individual_total:.2f}ms")
                logger.info(f"    Integrado: {integrated:.2f}ms")
                logger.info(f"    Overhead: {overhead:.2f}ms ({overhead_percent:.1f}%)")
                logger.info(f"    Eficiencia: {efficiency:.2%}")
        
        # Comparación 2: Compression individual vs Compression+Cache
        if compression_individual and compression_cache_integrated:
            if compression_individual.latency_ms > 0 and compression_cache_integrated.latency_ms > 0:
                individual_total = compression_individual.latency_ms
                integrated = compression_cache_integrated.latency_ms
                overhead = integrated - individual_total
                overhead_percent = (overhead / individual_total) * 100 if individual_total > 0 else 0
                speedup = individual_total / integrated if integrated > 0 else 0
                efficiency = individual_total / integrated if integrated > 0 else 0
                
                comparison = IntegrationComparison(
                    individual_total_ms=individual_total,
                    integrated_ms=integrated,
                    speedup=speedup,
                    overhead_ms=overhead,
                    overhead_percent=overhead_percent,
                    efficiency=efficiency
                )
                comparisons.append(comparison)
                
                logger.info(f"\n  Comparación Compression vs Compression+Cache:")
                logger.info(f"    Individual: {individual_total:.2f}ms")
                logger.info(f"    Integrado: {integrated:.2f}ms")
                logger.info(f"    Overhead: {overhead:.2f}ms ({overhead_percent:.1f}%)")
                logger.info(f"    Eficiencia: {efficiency:.2%}")
        
        # Comparación 3: Todos individuales vs Pipeline completo
        if attention_individual and compression_individual and full_pipeline:
            if (attention_individual.latency_ms > 0 and compression_individual.latency_ms > 0 
                and full_pipeline.latency_ms > 0):
                # Suma de tiempos individuales (simulado)
                individual_total = attention_individual.latency_ms + (compression_individual.latency_ms / 10)  # Ajustar escala
                integrated = full_pipeline.latency_ms
                overhead = integrated - individual_total
                overhead_percent = (overhead / individual_total) * 100 if individual_total > 0 else 0
                speedup = individual_total / integrated if integrated > 0 else 0
                efficiency = individual_total / integrated if integrated > 0 else 0
                
                comparison = IntegrationComparison(
                    individual_total_ms=individual_total,
                    integrated_ms=integrated,
                    speedup=speedup,
                    overhead_ms=overhead,
                    overhead_percent=overhead_percent,
                    efficiency=efficiency
                )
                comparisons.append(comparison)
                
                logger.info(f"\n  Comparación Individual vs Pipeline Completo:")
                logger.info(f"    Individual (suma): {individual_total:.2f}ms")
                logger.info(f"    Pipeline completo: {integrated:.2f}ms")
                logger.info(f"    Overhead: {overhead:.2f}ms ({overhead_percent:.1f}%)")
                logger.info(f"    Eficiencia: {efficiency:.2%}")
        
        self.comparisons = comparisons
        return comparisons
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Ejecutar todos los benchmarks."""
        logger.info("=" * 80)
        logger.info("BENCHMARK DE RENDIMIENTO - FRAMEWORK INTEGRADO")
        logger.info("=" * 80)
        
        # Benchmarks individuales
        logger.info("\n[Benchmarks Individuales]")
        self.results.append(self.benchmark_individual_attention())
        self.results.append(self.benchmark_individual_compression())
        
        # Benchmarks integrados
        logger.info("\n[Benchmarks Integrados]")
        self.results.append(self.benchmark_integrated_attention_cache())
        self.results.append(self.benchmark_integrated_compression_cache())
        self.results.append(self.benchmark_full_pipeline())
        
        # Comparaciones
        logger.info("\n[Comparaciones]")
        comparisons = self.compare_individual_vs_integrated()
        
        # Generar reporte
        report = {
            'timestamp': datetime.now().isoformat(),
            'individual_results': [r.to_dict() for r in self.results if 'individual' in r.name],
            'integrated_results': [r.to_dict() for r in self.results if 'integrated' in r.name or r.name == 'full_pipeline'],
            'comparisons': [c.to_dict() for c in comparisons],
            'summary': self.generate_summary(comparisons)
        }
        
        return report
    
    def generate_summary(self, comparisons: List[IntegrationComparison]) -> Dict[str, Any]:
        """Generar resumen de resultados."""
        if not comparisons:
            return {}
        
        avg_overhead = np.mean([c.overhead_percent for c in comparisons]) if comparisons else 0
        avg_efficiency = np.mean([c.efficiency for c in comparisons]) if comparisons else 0
        avg_speedup = np.mean([c.speedup for c in comparisons]) if comparisons else 0
        
        return {
            'average_overhead_percent': avg_overhead,
            'average_efficiency': avg_efficiency,
            'average_speedup': avg_speedup,
            'total_comparisons': len(comparisons)
        }
    
    def save_report(self, report: Dict[str, Any], path: Path):
        """Guardar reporte en JSON."""
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nReporte guardado en: {path}")


def print_report(report: Dict[str, Any]):
    """Imprimir reporte en consola."""
    print("\n" + "=" * 80)
    print("REPORTE DE RENDIMIENTO - FRAMEWORK INTEGRADO")
    print("=" * 80)
    print(f"Timestamp: {report['timestamp']}")
    
    print("\n[Resultados Individuales]")
    for result in report['individual_results']:
        if result['latency_ms'] > 0:
            print(f"  {result['name']}: {result['latency_ms']:.2f}ms, {result['throughput']:.2f} {result.get('metadata', {}).get('unit', 'ops/s')}")
    
    print("\n[Resultados Integrados]")
    for result in report['integrated_results']:
        if result['latency_ms'] > 0:
            print(f"  {result['name']}: {result['latency_ms']:.2f}ms, {result['throughput']:.2f} ops/s")
    
    print("\n[Comparaciones]")
    for i, comp in enumerate(report['comparisons'], 1):
        print(f"\n  Comparación {i}:")
        print(f"    Individual: {comp['individual_total_ms']:.2f}ms")
        print(f"    Integrado: {comp['integrated_ms']:.2f}ms")
        print(f"    Overhead: {comp['overhead_ms']:.2f}ms ({comp['overhead_percent']:.1f}%)")
        print(f"    Eficiencia: {comp['efficiency']:.2%}")
        if comp['speedup'] > 1:
            print(f"    Speedup: {comp['speedup']:.2f}x más rápido")
        elif comp['speedup'] < 1:
            print(f"    Slowdown: {1/comp['speedup']:.2f}x más lento")
    
    if report['summary']:
        summary = report['summary']
        print("\n[Resumen]")
        print(f"  Overhead promedio: {summary.get('average_overhead_percent', 0):.1f}%")
        print(f"  Eficiencia promedio: {summary.get('average_efficiency', 0):.2%}")
        print(f"  Speedup promedio: {summary.get('average_speedup', 0):.2f}x")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Framework Integration Performance Benchmark")
    parser.add_argument("--detailed", action="store_true", help="Mostrar detalles adicionales")
    parser.add_argument("--output", type=str, help="Directorio de salida para reportes")
    
    args = parser.parse_args()
    
    benchmark = FrameworkPerformanceBenchmark()
    report = benchmark.run_all_benchmarks()
    
    # Imprimir reporte
    print_report(report)
    
    # Guardar reporte
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "performance_reports"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / f"framework_integration_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    benchmark.save_report(report, report_path)
    
    logger.info(f"✓ Benchmark completado")


