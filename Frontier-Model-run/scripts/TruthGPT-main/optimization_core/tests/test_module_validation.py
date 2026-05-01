"""
Tests de validación para cada módulo del polyglot.

Estos tests verifican que cada módulo funcione correctamente antes de ejecutar benchmarks.
"""

import sys
import pytest
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRustBackend:
    """Tests para el backend Rust."""
    
    def test_rust_backend_import(self):
        """Verificar que el backend Rust se puede importar."""
        try:
            import truthgpt_rust
            assert True
        except ImportError:
            pytest.skip("Rust backend not available")
    
    def test_rust_kv_cache(self):
        """Test básico de KV Cache en Rust."""
        try:
            import truthgpt_rust
            cache = truthgpt_rust.PyKVCache(max_size=100)
            
            # Test PUT
            cache.put(0, 0, b"test_data")
            
            # Test GET
            data = cache.get(0, 0)
            assert data == b"test_data"
            
            # Test contains
            assert cache.contains(0, 0)
            assert not cache.contains(0, 1)
            
        except ImportError:
            pytest.skip("Rust backend not available")
    
    def test_rust_compression(self):
        """Test básico de Compression en Rust."""
        try:
            import truthgpt_rust
            compressor = truthgpt_rust.PyCompressor(algorithm="lz4", level=3)
            
            test_data = b"test_data_for_compression" * 100
            
            # Compress
            compressed = compressor.compress(test_data)
            assert len(compressed) < len(test_data)
            
            # Decompress
            decompressed = compressor.decompress(compressed)
            assert decompressed == test_data
            
        except ImportError:
            pytest.skip("Rust backend not available")


class TestCppBackend:
    """Tests para el backend C++."""
    
    def test_cpp_backend_import(self):
        """Verificar que el backend C++ se puede importar."""
        try:
            import _cpp_core
            assert True
        except ImportError:
            pytest.skip("C++ backend not available")
    
    def test_cpp_attention(self):
        """Test básico de Attention en C++."""
        try:
            import _cpp_core
            import numpy as np
            
            # Verificar que el módulo tiene attention
            if hasattr(_cpp_core, 'attention'):
                attn = _cpp_core.attention
                # Test básico (depende de la implementación)
                assert attn is not None
        except ImportError:
            pytest.skip("C++ backend not available")
        except AttributeError:
            pytest.skip("C++ attention module not available")


class TestGoBackend:
    """Tests para el backend Go."""
    
    def test_go_backend_import(self):
        """Verificar que el backend Go se puede importar."""
        try:
            from optimization_core.go_core.pkg.client import python_client
            assert True
        except ImportError:
            pytest.skip("Go backend not available")
    
    def test_go_client(self):
        """Test básico del cliente Go."""
        try:
            from optimization_core.go_core.pkg.client import python_client
            
            # Verificar que el cliente existe
            assert python_client is not None
            
        except ImportError:
            pytest.skip("Go backend not available")


class TestPolyglotCore:
    """Tests para polyglot_core."""
    
    def test_polyglot_core_import(self):
        """Verificar que polyglot_core se puede importar."""
        try:
            from optimization_core.polyglot_core import (
                Backend, get_available_backends, is_backend_available
            )
            assert True
        except ImportError:
            pytest.skip("polyglot_core not available")
    
    def test_backend_detection(self):
        """Test de detección de backends."""
        try:
            from optimization_core.polyglot_core import (
                get_available_backends, is_backend_available
            )
            
            backends = get_available_backends()
            assert len(backends) > 0  # Al menos Python debe estar disponible
            
            # Python siempre debe estar disponible
            assert is_backend_available('python')
            
        except ImportError:
            pytest.skip("polyglot_core not available")
    
    def test_unified_kv_cache(self):
        """Test de UnifiedKVCache."""
        try:
            from optimization_core.polyglot_core import UnifiedKVCache, Backend
            
            # Test con auto-selection
            cache = UnifiedKVCache(max_size=100)
            
            # Test PUT
            cache.put(0, 0, b"test_data")
            
            # Test GET
            data = cache.get(0, 0)
            assert data == b"test_data"
            
            # Test contains
            assert cache.contains(0, 0)
            assert not cache.contains(0, 1)
            
        except ImportError:
            pytest.skip("polyglot_core not available")
    
    def test_unified_compressor(self):
        """Test de UnifiedCompressor."""
        try:
            from optimization_core.polyglot_core import UnifiedCompressor, Backend
            
            compressor = UnifiedCompressor(algorithm="lz4")
            
            test_data = b"test_data_for_compression" * 100
            
            # Compress
            compressed = compressor.compress(test_data)
            assert len(compressed) < len(test_data)
            
            # Decompress
            decompressed = compressor.decompress(compressed)
            assert decompressed == test_data
            
        except ImportError:
            pytest.skip("polyglot_core not available")


class TestInferenceEngines:
    """Tests para engines de inferencia."""
    
    def test_engine_factory_import(self):
        """Verificar que engine_factory se puede importar."""
        try:
            from optimization_core.inference.engine_factory import (
                create_inference_engine, list_available_engines
            )
            assert True
        except ImportError:
            pytest.skip("inference engines not available")
    
    def test_list_available_engines(self):
        """Test de listado de engines disponibles."""
        try:
            from optimization_core.inference.engine_factory import list_available_engines
            
            engines = list_available_engines()
            assert isinstance(engines, dict)
            
        except ImportError:
            pytest.skip("inference engines not available")
    
    def test_vllm_engine_import(self):
        """Verificar que vLLM engine se puede importar."""
        try:
            from optimization_core.inference.vllm_engine import VLLMEngine, VLLM_AVAILABLE
            # No necesitamos crear el engine, solo verificar que se puede importar
            assert True
        except ImportError:
            pytest.skip("vLLM engine not available")
    
    def test_tensorrt_engine_import(self):
        """Verificar que TensorRT-LLM engine se puede importar."""
        try:
            from optimization_core.inference.tensorrt_llm_engine import (
                TensorRTLLMEngine, TENSORRT_LLM_AVAILABLE
            )
            # No necesitamos crear el engine, solo verificar que se puede importar
            assert True
        except ImportError:
            pytest.skip("TensorRT-LLM engine not available")


class TestAttention:
    """Tests para el módulo de Attention."""
    
    def test_attention_import(self):
        """Verificar que Attention se puede importar."""
        try:
            from optimization_core.polyglot_core.attention import Attention
            assert True
        except ImportError:
            pytest.skip("Attention module not available")
    
    def test_attention_forward(self):
        """Test básico de forward pass de Attention."""
        try:
            from optimization_core.polyglot_core.attention import Attention, AttentionConfig
            import numpy as np
            
            config = AttentionConfig(
                d_model=768,
                n_heads=12,
                d_k=64
            )
            
            attention = Attention(config)
            
            batch_size = 2
            seq_len = 128
            d_model = 768
            
            q = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            k = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            v = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
            
            output = attention.forward(q, k, v)
            
            assert output.shape == (batch_size, seq_len, d_model)
            
        except ImportError:
            pytest.skip("Attention module not available")


class TestCompression:
    """Tests para el módulo de Compression."""
    
    def test_compression_import(self):
        """Verificar que Compression se puede importar."""
        try:
            from optimization_core.polyglot_core.compression import Compressor
            assert True
        except ImportError:
            pytest.skip("Compression module not available")


class TestCache:
    """Tests para el módulo de Cache."""
    
    def test_cache_import(self):
        """Verificar que Cache se puede importar."""
        try:
            from optimization_core.polyglot_core.cache import KVCache
            assert True
        except ImportError:
            pytest.skip("Cache module not available")


def test_all_modules_summary():
    """Test que genera un resumen de todos los módulos disponibles."""
    modules_status = {}
    
    # Rust
    try:
        import truthgpt_rust
        modules_status['rust'] = True
    except ImportError:
        modules_status['rust'] = False
    
    # C++
    try:
        import _cpp_core
        modules_status['cpp'] = True
    except ImportError:
        modules_status['cpp'] = False
    
    # Go
    try:
        from optimization_core.go_core.pkg.client import python_client
        modules_status['go'] = True
    except ImportError:
        modules_status['go'] = False
    
    # Polyglot Core
    try:
        from optimization_core.polyglot_core import get_available_backends
        modules_status['polyglot_core'] = True
    except ImportError:
        modules_status['polyglot_core'] = False
    
    # Inference Engines
    try:
        from optimization_core.inference.engine_factory import list_available_engines
        modules_status['inference_engines'] = True
    except ImportError:
        modules_status['inference_engines'] = False
    
    # Log summary
    logger.info("=" * 60)
    logger.info("RESUMEN DE MÓDULOS DISPONIBLES")
    logger.info("=" * 60)
    for module, available in modules_status.items():
        status = "✓" if available else "✗"
        logger.info(f"{status} {module}")
    
    available_count = sum(1 for v in modules_status.values() if v)
    total_count = len(modules_status)
    logger.info(f"\nTotal: {available_count}/{total_count} módulos disponibles")
    
    # Al menos algunos módulos deben estar disponibles
    assert available_count > 0, "Ningún módulo está disponible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])













