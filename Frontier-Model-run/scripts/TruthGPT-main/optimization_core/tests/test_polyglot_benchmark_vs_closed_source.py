"""
Test completo de benchmark del modelo Polyglot vs modelos closed source.

Este test:
1. Prueba todos los módulos del polyglot (Rust, C++, Go, Python)
2. Compara velocidad con modelos closed source (GPT-4, Claude, etc.)
3. Valida que todos los módulos funcionen correctamente
4. Genera reportes detallados de rendimiento

Usage:
    python -m pytest tests/test_polyglot_benchmark_vs_closed_source.py -v
    python tests/test_polyglot_benchmark_vs_closed_source.py --full
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
# Also add current directory for direct imports
sys.path.insert(0, str(parent_dir.parent) if parent_dir.parent.exists() else str(parent_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import all polyglot modules
POLYGLOT_MODULES = {}

try:
    # Try direct import first
    try:
        from polyglot_core import (
            Backend, BackendInfo, get_available_backends,
            get_best_backend, is_backend_available,
            UnifiedKVCache, UnifiedCompressor
        )
    except ImportError:
        from optimization_core.polyglot_core import (
            Backend, BackendInfo, get_available_backends,
            get_best_backend, is_backend_available,
            UnifiedKVCache, UnifiedCompressor
        )
    POLYGLOT_MODULES['polyglot_core'] = True
except ImportError as e:
    logger.warning(f"polyglot_core not available: {e}")
    POLYGLOT_MODULES['polyglot_core'] = False

try:
    try:
        from polyglot_core.inference import (
            InferenceEngine, GenerationConfig, InferenceConfig
        )
    except ImportError:
        from optimization_core.polyglot_core.inference import (
            InferenceEngine, GenerationConfig, InferenceConfig
        )
    POLYGLOT_MODULES['polyglot_inference'] = True
except ImportError as e:
    logger.warning(f"polyglot_inference not available: {e}")
    POLYGLOT_MODULES['polyglot_inference'] = False

try:
    try:
        from inference.engine_factory import (
            create_inference_engine, EngineType, list_available_engines
        )
    except ImportError:
        from optimization_core.inference.engine_factory import (
            create_inference_engine, EngineType, list_available_engines
        )
    POLYGLOT_MODULES['inference_engines'] = True
except ImportError as e:
    logger.warning(f"inference_engines not available: {e}")
    POLYGLOT_MODULES['inference_engines'] = False

try:
    try:
        from polyglot_core.attention import Attention
    except ImportError:
        from optimization_core.polyglot_core.attention import Attention
    POLYGLOT_MODULES['attention'] = True
except ImportError as e:
    logger.warning(f"attention not available: {e}")
    POLYGLOT_MODULES['attention'] = False

try:
    try:
        from polyglot_core.compression import Compressor
    except ImportError:
        from optimization_core.polyglot_core.compression import Compressor
    POLYGLOT_MODULES['compression'] = True
except ImportError as e:
    logger.warning(f"compression not available: {e}")
    POLYGLOT_MODULES['compression'] = False

try:
    try:
        from polyglot_core.cache import KVCache
    except ImportError:
        from optimization_core.polyglot_core.cache import KVCache
    POLYGLOT_MODULES['cache'] = True
except ImportError as e:
    logger.warning(f"cache not available: {e}")
    POLYGLOT_MODULES['cache'] = False

# Try to import Rust backend
try:
    import truthgpt_rust
    POLYGLOT_MODULES['rust'] = True
except ImportError:
    POLYGLOT_MODULES['rust'] = False
    logger.warning("Rust backend not available")

# Try to import C++ backend
try:
    import _cpp_core
    POLYGLOT_MODULES['cpp'] = True
except ImportError:
    POLYGLOT_MODULES['cpp'] = False
    logger.warning("C++ backend not available")

# Try to import Go backend
try:
    from optimization_core.go_core.pkg.client import python_client
    POLYGLOT_MODULES['go'] = True
except ImportError:
    POLYGLOT_MODULES['go'] = False
    logger.warning("Go backend not available")


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    name: str
    module: str
    backend: str
    latency_ms: float
    throughput_tokens_per_sec: float = 0.0
    memory_mb: float = 0.0
    tokens_generated: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedSourceResult:
    """Resultado de un modelo closed source."""
    model_name: str
    latency_ms: float
    tokens_generated: int
    cost_per_1k_tokens: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    
    @property
    def throughput_tokens_per_sec(self) -> float:
        if self.latency_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.latency_ms / 1000.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'throughput_tokens_per_sec': self.throughput_tokens_per_sec
        }


@dataclass
class BenchmarkReport:
    """Reporte completo de benchmarks."""
    timestamp: str
    polyglot_results: List[BenchmarkResult]
    closed_source_results: List[ClosedSourceResult]
    module_status: Dict[str, bool]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'polyglot_results': [r.to_dict() for r in self.polyglot_results],
            'closed_source_results': [r.to_dict() for r in self.closed_source_results],
            'module_status': self.module_status,
            'summary': self.summary
        }
    
    def save_json(self, path: Path):
        """Guardar reporte en JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Reporte guardado en: {path}")


class PolyglotBenchmarker:
    """Benchmarker para el modelo polyglot completo."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the main differences between Rust and Python?",
            "Describe the architecture of a transformer model.",
            "How does attention mechanism work in neural networks?",
        ]
    
    def benchmark_kv_cache(self) -> List[BenchmarkResult]:
        """Benchmark KV Cache en todos los backends disponibles."""
        results = []
        logger.info("Benchmarking KV Cache...")
        
        if not POLYGLOT_MODULES.get('cache', False):
            logger.warning("KV Cache module not available")
            return results
        
        try:
            try:
                try:
                    from polyglot_core import UnifiedKVCache, Backend, is_backend_available
                except ImportError:
                    from optimization_core.polyglot_core import UnifiedKVCache, Backend, is_backend_available
            except ImportError as e:
                logger.error(f"Failed to import KV Cache modules: {e}")
                return results
            
            # Test con diferentes backends
            backends_to_test = []
            try:
                if is_backend_available('rust'):
                    backends_to_test.append(('rust', Backend.RUST))
                if is_backend_available('cpp'):
                    backends_to_test.append(('cpp', Backend.CPP))
                if is_backend_available('go'):
                    backends_to_test.append(('go', Backend.GO))
            except:
                pass
            backends_to_test.append(('python', Backend.PYTHON))
            
            for backend_name, backend in backends_to_test:
                try:
                    cache = UnifiedKVCache(max_size=8192, backend=backend)
                    
                    # Create test tensors
                    import numpy as np
                    test_key = np.random.randn(768).astype(np.float32)
                    test_value = np.random.randn(768).astype(np.float32)
                    
                    # Warmup
                    for i in range(10):
                        cache.put(layer=0, position=i, key=test_key, value=test_value)
                    
                    # Benchmark PUT operations
                    start = time.perf_counter()
                    iterations = 1000
                    for i in range(iterations):
                        cache.put(layer=0, position=i % 100, key=test_key, value=test_value)
                    put_time_ms = (time.perf_counter() - start) * 1000
                    
                    # Benchmark GET operations
                    start = time.perf_counter()
                    for i in range(iterations):
                        cache.get(layer=0, position=i % 100)
                    get_time_ms = (time.perf_counter() - start) * 1000
                    
                    results.append(BenchmarkResult(
                        name="kv_cache_put",
                        module="kv_cache",
                        backend=backend_name,
                        latency_ms=put_time_ms / iterations,
                        throughput_tokens_per_sec=iterations / (put_time_ms / 1000),
                        success=True,
                        metadata={'operations': iterations}
                    ))
                    
                    results.append(BenchmarkResult(
                        name="kv_cache_get",
                        module="kv_cache",
                        backend=backend_name,
                        latency_ms=get_time_ms / iterations,
                        throughput_tokens_per_sec=iterations / (get_time_ms / 1000),
                        success=True,
                        metadata={'operations': iterations}
                    ))
                    
                    logger.info(f"  [OK] KV Cache ({backend_name}): PUT={put_time_ms/iterations:.3f}ms, GET={get_time_ms/iterations:.3f}ms")
                    
                except Exception as e:
                    logger.error(f"  [X] KV Cache ({backend_name}) failed: {e}")
                    results.append(BenchmarkResult(
                        name=f"kv_cache_{backend_name}",
                        module="kv_cache",
                        backend=backend_name,
                        latency_ms=0.0,
                        success=False,
                        error=str(e)
                    ))
        
        except Exception as e:
            logger.error(f"KV Cache benchmark failed: {e}")
        
        return results
    
    def benchmark_compression(self) -> List[BenchmarkResult]:
        """Benchmark Compression en todos los backends."""
        results = []
        logger.info("Benchmarking Compression...")
        
        if not POLYGLOT_MODULES.get('compression', False):
            logger.warning("Compression module not available")
            return results
        
        try:
            try:
                from polyglot_core import UnifiedCompressor, Backend, is_backend_available
            except ImportError:
                from optimization_core.polyglot_core import UnifiedCompressor, Backend, is_backend_available
        except ImportError as e:
            logger.error(f"Failed to import Compression modules: {e}")
            return results
        
        test_data = b"test_data_for_compression" * 1000  # ~25KB
        
        backends_to_test = []
        try:
            if is_backend_available('rust'):
                backends_to_test.append(('rust', Backend.RUST))
            if is_backend_available('cpp'):
                backends_to_test.append(('cpp', Backend.CPP))
            if is_backend_available('go'):
                backends_to_test.append(('go', Backend.GO))
        except:
            pass
        backends_to_test.append(('python', Backend.PYTHON))
        
        for backend_name, backend in backends_to_test:
            try:
                compressor = UnifiedCompressor(algorithm="lz4", backend=backend)
                    
                # Warmup
                for _ in range(5):
                    result = compressor.compress(test_data)
                    decompressed = compressor.decompress(result.data)
                
                # Benchmark compression
                start = time.perf_counter()
                iterations = 100
                compressed_result = None
                for _ in range(iterations):
                    compressed_result = compressor.compress(test_data)
                compress_time_ms = (time.perf_counter() - start) * 1000
                
                # Benchmark decompression
                start = time.perf_counter()
                for _ in range(iterations):
                    decompressed = compressor.decompress(compressed_result.data)
                decompress_time_ms = (time.perf_counter() - start) * 1000
                
                compression_ratio = compressed_result.stats.compression_ratio if compressed_result.stats else len(compressed_result.data) / len(test_data)
                
                results.append(BenchmarkResult(
                    name="compression_compress",
                    module="compression",
                    backend=backend_name,
                    latency_ms=compress_time_ms / iterations,
                    throughput_tokens_per_sec=(len(test_data) * iterations) / (compress_time_ms / 1000) / 1024 / 1024,  # MB/s
                    success=True,
                    metadata={
                        'data_size_bytes': len(test_data),
                        'compressed_size_bytes': len(compressed_result.data) if compressed_result else 0,
                        'compression_ratio': compression_ratio
                    }
                ))
                
                results.append(BenchmarkResult(
                    name="compression_decompress",
                    module="compression",
                    backend=backend_name,
                    latency_ms=decompress_time_ms / iterations,
                    throughput_tokens_per_sec=(len(test_data) * iterations) / (decompress_time_ms / 1000) / 1024 / 1024,  # MB/s
                    success=True,
                    metadata={'data_size_bytes': len(test_data)}
                ))
                
                logger.info(f"  [OK] Compression ({backend_name}): Compress={compress_time_ms/iterations:.3f}ms, Decompress={decompress_time_ms/iterations:.3f}ms")
                
            except Exception as e:
                logger.error(f"  [X] Compression ({backend_name}) failed: {e}")
                results.append(BenchmarkResult(
                    name=f"compression_{backend_name}",
                    module="compression",
                    backend=backend_name,
                    latency_ms=0.0,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def benchmark_attention(self) -> List[BenchmarkResult]:
        """Benchmark Attention en todos los backends."""
        results = []
        logger.info("Benchmarking Attention...")
        
        if not POLYGLOT_MODULES.get('attention', False):
            logger.warning("Attention module not available")
            return results
        
        try:
            try:
                from polyglot_core.attention import Attention, AttentionConfig
                from polyglot_core import Backend, is_backend_available
            except ImportError:
                from optimization_core.polyglot_core.attention import Attention, AttentionConfig
                from optimization_core.polyglot_core import Backend, is_backend_available
        except ImportError as e:
            logger.error(f"Failed to import Attention modules: {e}")
            return results
        
        batch_size = 4
        seq_len = 512
        d_model = 768
        n_heads = 12
        
        config = AttentionConfig(
            d_model=d_model,
            n_heads=n_heads
        )
        
        # Generate random inputs - flatten for attention API
        q_flat = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
        k_flat = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
        v_flat = np.random.randn(batch_size * seq_len, d_model).astype(np.float32)
        
        backends_to_test = []
        try:
            if is_backend_available('cpp'):
                backends_to_test.append(('cpp', Backend.CPP))
            if is_backend_available('rust'):
                backends_to_test.append(('rust', Backend.RUST))
        except:
            pass
        backends_to_test.append(('python', Backend.PYTHON))
        
        for backend_name, backend in backends_to_test:
            try:
                attention = Attention(config, backend=backend)
                
                # Warmup
                for _ in range(3):
                    _ = attention.forward(q_flat, k_flat, v_flat, batch_size, seq_len)
                
                # Benchmark
                start = time.perf_counter()
                iterations = 10
                for _ in range(iterations):
                    output = attention.forward(q_flat, k_flat, v_flat, batch_size, seq_len)
                total_time_ms = (time.perf_counter() - start) * 1000
                
                avg_latency_ms = total_time_ms / iterations
                tokens_per_sec = (batch_size * seq_len * iterations) / (total_time_ms / 1000)
                
                results.append(BenchmarkResult(
                    name="attention_forward",
                    module="attention",
                    backend=backend_name,
                    latency_ms=avg_latency_ms,
                    throughput_tokens_per_sec=tokens_per_sec,
                    success=True,
                    metadata={
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'd_model': d_model,
                        'n_heads': n_heads
                    }
                ))
                
                logger.info(f"  [OK] Attention ({backend_name}): {avg_latency_ms:.2f}ms, {tokens_per_sec:.0f} tokens/s")
                
            except Exception as e:
                logger.error(f"  [X] Attention ({backend_name}) failed: {e}")
                results.append(BenchmarkResult(
                    name=f"attention_{backend_name}",
                    module="attention",
                    backend=backend_name,
                    latency_ms=0.0,
                    success=False,
                    error=str(e)
                    ))
        
        return results
    
    def benchmark_inference(self, model_path: Optional[str] = None) -> List[BenchmarkResult]:
        """Benchmark Inference engines."""
        results = []
        logger.info("Benchmarking Inference Engines...")
        
        if not POLYGLOT_MODULES.get('inference_engines', False):
            logger.warning("Inference engines not available")
            return results
        
        try:
            try:
                from inference.engine_factory import (
                    create_inference_engine, EngineType, list_available_engines
                )
            except ImportError:
                from optimization_core.inference.engine_factory import (
                    create_inference_engine, EngineType, list_available_engines
                )
            
            available_engines = list_available_engines()
            logger.info(f"Available engines: {available_engines}")
            
            # Use a small test model if available, otherwise skip
            test_model = model_path or "gpt2"  # Small model for testing
            
            for engine_type in [EngineType.VLLM, EngineType.TENSORRT_LLM]:
                if not available_engines.get(engine_type, False):
                    continue
                
                try:
                    logger.info(f"  Testing {engine_type}...")
                    engine = create_inference_engine(
                        model=test_model,
                        engine_type=engine_type,
                        prefer_gpu=True
                    )
                    
                    # Test with first prompt
                    prompt = self.test_prompts[0]
                    
                    # Warmup
                    try:
                        _ = engine.generate(
                            prompts=[prompt],
                            max_new_tokens=10,
                            temperature=0.7
                        )
                    except Exception as e:
                        logger.warning(f"  Warmup failed for {engine_type}: {e}")
                        continue
                    
                    # Benchmark
                    start = time.perf_counter()
                    iterations = 5
                    total_tokens = 0
                    
                    for i in range(iterations):
                        prompt = self.test_prompts[i % len(self.test_prompts)]
                        try:
                            result = engine.generate(
                                prompts=[prompt],
                                max_new_tokens=50,
                                temperature=0.7
                            )
                            if hasattr(result, 'tokens_generated'):
                                total_tokens += result.tokens_generated
                            elif isinstance(result, list) and len(result) > 0:
                                # Estimate tokens (rough)
                                total_tokens += len(result[0].split()) * 1.3  # ~1.3 tokens per word
                        except Exception as e:
                            logger.warning(f"  Generation {i} failed: {e}")
                    
                    total_time_ms = (time.perf_counter() - start) * 1000
                    avg_latency_ms = total_time_ms / iterations
                    tokens_per_sec = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
                    
                    results.append(BenchmarkResult(
                        name=f"inference_{engine_type}",
                        module="inference",
                        backend=engine_type,
                        latency_ms=avg_latency_ms,
                        throughput_tokens_per_sec=tokens_per_sec,
                        tokens_generated=total_tokens,
                        success=True,
                        metadata={'model': test_model, 'iterations': iterations}
                    ))
                    
                    logger.info(f"  [OK] Inference ({engine_type}): {avg_latency_ms:.2f}ms, {tokens_per_sec:.1f} tokens/s")
                    
                except Exception as e:
                    logger.error(f"  [X] Inference ({engine_type}) failed: {e}")
                    results.append(BenchmarkResult(
                        name=f"inference_{engine_type}",
                        module="inference",
                        backend=engine_type,
                        latency_ms=0.0,
                        success=False,
                        error=str(e)
                    ))
        
        except Exception as e:
            logger.error(f"Inference benchmark failed: {e}")
        
        return results
    
    def run_all_benchmarks(self, model_path: Optional[str] = None) -> List[BenchmarkResult]:
        """Ejecutar todos los benchmarks."""
        logger.info("=" * 80)
        logger.info("Ejecutando benchmarks completos del modelo Polyglot")
        logger.info("=" * 80)
        
        all_results = []
        
        # Benchmark cada módulo
        all_results.extend(self.benchmark_kv_cache())
        all_results.extend(self.benchmark_compression())
        all_results.extend(self.benchmark_attention())
        all_results.extend(self.benchmark_inference(model_path))
        
        self.results = all_results
        return all_results


class ClosedSourceBenchmarker:
    """Benchmarker para modelos closed source (GPT-4, Claude, etc.)."""
    
    def __init__(self):
        self.results: List[ClosedSourceResult] = []
        self.test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the main differences between Rust and Python?",
            "Describe the architecture of a transformer model.",
            "How does attention mechanism work in neural networks?",
        ]
    
    def benchmark_openai(self, api_key: Optional[str] = None) -> List[ClosedSourceResult]:
        """Benchmark OpenAI GPT-4."""
        results = []
        logger.info("Benchmarking OpenAI GPT-4...")
        
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("  OpenAI API key not found, skipping")
            return results
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            for model_name in ['gpt-4', 'gpt-3.5-turbo']:
                try:
                    total_time_ms = 0
                    total_tokens = 0
                    iterations = 3
                    
                    for i in range(iterations):
                        prompt = self.test_prompts[i % len(self.test_prompts)]
                        
                        start = time.perf_counter()
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=50,
                            temperature=0.7
                        )
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        
                        total_time_ms += elapsed_ms
                        total_tokens += response.usage.completion_tokens if response.usage else 50
                    
                    avg_latency_ms = total_time_ms / iterations
                    
                    # Cost estimates (as of 2024)
                    cost_per_1k = {
                        'gpt-4': 0.03,  # $0.03 per 1K tokens
                        'gpt-3.5-turbo': 0.002  # $0.002 per 1K tokens
                    }.get(model_name, 0.01)
                    
                    results.append(ClosedSourceResult(
                        model_name=model_name,
                        latency_ms=avg_latency_ms,
                        tokens_generated=total_tokens,
                        cost_per_1k_tokens=cost_per_1k,
                        success=True
                    ))
                    
                    logger.info(f"  ✓ {model_name}: {avg_latency_ms:.2f}ms, {results[-1].throughput_tokens_per_sec:.1f} tokens/s")
                    
                except Exception as e:
                    logger.error(f"  ✗ {model_name} failed: {e}")
                    results.append(ClosedSourceResult(
                        model_name=model_name,
                        latency_ms=0.0,
                        tokens_generated=0,
                        success=False,
                        error=str(e)
                    ))
        
        except ImportError:
            logger.warning("  openai library not installed, skipping")
        except Exception as e:
            logger.error(f"OpenAI benchmark failed: {e}")
        
        return results
    
    def benchmark_anthropic(self, api_key: Optional[str] = None) -> List[ClosedSourceResult]:
        """Benchmark Anthropic Claude."""
        results = []
        logger.info("Benchmarking Anthropic Claude...")
        
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("  Anthropic API key not found, skipping")
            return results
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            for model_name in ['claude-3-opus-20240229', 'claude-3-sonnet-20240229']:
                try:
                    total_time_ms = 0
                    total_tokens = 0
                    iterations = 3
                    
                    for i in range(iterations):
                        prompt = self.test_prompts[i % len(self.test_prompts)]
                        
                        start = time.perf_counter()
                        message = client.messages.create(
                            model=model_name,
                            max_tokens=50,
                            temperature=0.7,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        
                        total_time_ms += elapsed_ms
                        total_tokens += message.usage.output_tokens if message.usage else 50
                    
                    avg_latency_ms = total_time_ms / iterations
                    
                    # Cost estimates
                    cost_per_1k = {
                        'claude-3-opus-20240229': 0.015,
                        'claude-3-sonnet-20240229': 0.003
                    }.get(model_name, 0.01)
                    
                    results.append(ClosedSourceResult(
                        model_name=model_name,
                        latency_ms=avg_latency_ms,
                        tokens_generated=total_tokens,
                        cost_per_1k_tokens=cost_per_1k,
                        success=True
                    ))
                    
                    logger.info(f"  ✓ {model_name}: {avg_latency_ms:.2f}ms, {results[-1].throughput_tokens_per_sec:.1f} tokens/s")
                    
                except Exception as e:
                    logger.error(f"  ✗ {model_name} failed: {e}")
                    results.append(ClosedSourceResult(
                        model_name=model_name,
                        latency_ms=0.0,
                        tokens_generated=0,
                        success=False,
                        error=str(e)
                    ))
        
        except ImportError:
            logger.warning("  anthropic library not installed, skipping")
        except Exception as e:
            logger.error(f"Anthropic benchmark failed: {e}")
        
        return results
    
    def run_all_benchmarks(self) -> List[ClosedSourceResult]:
        """Ejecutar todos los benchmarks de closed source."""
        logger.info("=" * 80)
        logger.info("Ejecutando benchmarks de modelos closed source")
        logger.info("=" * 80)
        
        all_results = []
        all_results.extend(self.benchmark_openai())
        all_results.extend(self.benchmark_anthropic())
        
        self.results = all_results
        return all_results


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
    print(f"\nEstado de Modulos:")
    for module, status in report.module_status.items():
        status_str = "[OK] Disponible" if status else "[X] No disponible"
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


# ============================================================================
# TESTS
# ============================================================================

def test_all_modules_available():
    """Test que verifica que los módulos principales estén disponibles."""
    logger.info("Verificando disponibilidad de módulos...")
    
    available_count = sum(1 for v in POLYGLOT_MODULES.values() if v)
    total_count = len(POLYGLOT_MODULES)
    
    logger.info(f"Módulos disponibles: {available_count}/{total_count}")
    for module, available in POLYGLOT_MODULES.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {module}")
    
    # Al menos algunos módulos deben estar disponibles
    assert available_count > 0, "Ningún módulo polyglot está disponible"


def test_polyglot_benchmarks():
    """Test principal de benchmarks polyglot."""
    benchmarker = PolyglotBenchmarker()
    results = benchmarker.run_all_benchmarks()
    
    # Verificar que al menos algunos benchmarks se ejecutaron
    assert len(results) > 0, "No se ejecutaron benchmarks"
    
    # Verificar que al menos algunos fueron exitosos
    successful = [r for r in results if r.success]
    assert len(successful) > 0, "Ningún benchmark fue exitoso"
    
    logger.info(f"Benchmarks ejecutados: {len(results)}, Exitosos: {len(successful)}")


def test_closed_source_benchmarks():
    """Test de benchmarks de modelos closed source."""
    benchmarker = ClosedSourceBenchmarker()
    results = benchmarker.run_all_benchmarks()
    
    # Este test puede fallar si no hay API keys, pero no es crítico
    if len(results) == 0:
        logger.warning("No se pudieron ejecutar benchmarks de closed source (probablemente falta API key)")
        if pytest:
            pytest.skip("API keys no disponibles")
        else:
            logger.warning("Skipping closed source benchmarks - API keys no disponibles")
    
    successful = [r for r in results if r.success]
    logger.info(f"Closed source benchmarks: {len(results)}, Exitosos: {len(successful)}")


def test_full_benchmark_suite():
    """Test completo que ejecuta todos los benchmarks y genera reporte."""
    # Ejecutar benchmarks polyglot
    polyglot_benchmarker = PolyglotBenchmarker()
    polyglot_results = polyglot_benchmarker.run_all_benchmarks()
    
    # Ejecutar benchmarks closed source
    closed_source_benchmarker = ClosedSourceBenchmarker()
    closed_source_results = closed_source_benchmarker.run_all_benchmarks()
    
    # Generar resumen
    summary = generate_summary(polyglot_results, closed_source_results)
    
    # Crear reporte
    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
        polyglot_results=polyglot_results,
        closed_source_results=closed_source_results,
        module_status=POLYGLOT_MODULES,
        summary=summary
    )
    
    # Imprimir reporte
    print_report(report)
    
    # Guardar reporte
    output_dir = Path(__file__).parent.parent / "benchmark_reports"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report.save_json(report_path)
    
    # Verificaciones
    assert len(polyglot_results) > 0, "No se ejecutaron benchmarks polyglot"
    successful_polyglot = sum(1 for r in polyglot_results if r.success)
    assert successful_polyglot > 0, "Ningún benchmark polyglot fue exitoso"
    
    logger.info(f"Reporte guardado en: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Polyglot vs Closed Source")
    parser.add_argument("--full", action="store_true", help="Ejecutar suite completa")
    parser.add_argument("--model", type=str, help="Ruta al modelo para inference")
    parser.add_argument("--output", type=str, help="Directorio de salida para reportes")
    
    args = parser.parse_args()
    
    if args.full:
        # Ejecutar suite completa
        test_full_benchmark_suite()
    else:
        # Ejecutar tests individuales
        test_all_modules_available()
        test_polyglot_benchmarks()
        test_closed_source_benchmarks()


