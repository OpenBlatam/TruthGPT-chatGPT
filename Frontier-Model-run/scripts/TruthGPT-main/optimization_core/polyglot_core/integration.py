"""
Integration utilities for polyglot_core.

Provides compatibility helpers and integration with existing test suites.
"""

from typing import Dict, List, Optional, Any, Union
import warnings


# Compatibility aliases for existing code
def get_unified_kvcache(*args, **kwargs):
    """
    Compatibility alias for UnifiedKVCache.
    
    Maps to KVCache for backward compatibility.
    """
    from .cache import KVCache
    warnings.warn(
        "UnifiedKVCache is deprecated, use KVCache instead",
        DeprecationWarning,
        stacklevel=2
    )
    return KVCache(*args, **kwargs)


def get_unified_compressor(*args, **kwargs):
    """
    Compatibility alias for UnifiedCompressor.
    
    Maps to Compressor for backward compatibility.
    """
    from .compression import Compressor
    warnings.warn(
        "UnifiedCompressor is deprecated, use Compressor instead",
        DeprecationWarning,
        stacklevel=2
    )
    return Compressor(*args, **kwargs)


# Export for backward compatibility
UnifiedKVCache = get_unified_kvcache
UnifiedCompressor = get_unified_compressor


def check_polyglot_availability() -> Dict[str, bool]:
    """
    Check availability of all polyglot modules.
    
    Returns:
        Dict mapping module name to availability
    """
    availability = {}
    
    # Check main module
    try:
        from . import backend
        availability['polyglot_core'] = True
    except ImportError:
        availability['polyglot_core'] = False
    
    # Check submodules
    modules = [
        'cache', 'attention', 'compression', 'inference',
        'tokenization', 'quantization', 'profiling', 'benchmarking'
    ]
    
    for module in modules:
        try:
            __import__(f'optimization_core.polyglot_core.{module}')
            availability[module] = True
        except ImportError:
            availability[module] = False
    
    # Check backends
    try:
        from .backend import get_available_backends
        backends = get_available_backends()
        for backend_info in backends:
            availability[f'backend_{backend_info.name}'] = backend_info.available
    except Exception:
        pass
    
    return availability


def print_polyglot_status():
    """Print status of all polyglot components."""
    availability = check_polyglot_availability()
    
    print("\n" + "=" * 70)
    print("Polyglot Core Status")
    print("=" * 70)
    
    # Main module
    if availability.get('polyglot_core', False):
        print("✓ polyglot_core: Available")
    else:
        print("✗ polyglot_core: Not available")
    
    # Submodules
    print("\nModules:")
    for module in ['cache', 'attention', 'compression', 'inference',
                   'tokenization', 'quantization', 'profiling', 'benchmarking']:
        status = "✓" if availability.get(module, False) else "✗"
        print(f"  {status} {module}")
    
    # Backends
    print("\nBackends:")
    for key, available in availability.items():
        if key.startswith('backend_'):
            backend_name = key.replace('backend_', '')
            status = "✓" if available else "✗"
            print(f"  {status} {backend_name}")
    
    print("=" * 70 + "\n")


def get_test_compatibility_info() -> Dict[str, Any]:
    """
    Get information for test compatibility.
    
    Returns:
        Dict with compatibility information
    """
    from .backend import get_available_backends, Backend
    
    info = {
        'polyglot_available': False,
        'backends': {},
        'modules': {},
        'recommendations': []
    }
    
    try:
        from . import backend
        info['polyglot_available'] = True
        
        # Backend info
        backends = get_available_backends()
        for backend_info in backends:
            info['backends'][backend_info.name] = {
                'available': backend_info.available,
                'version': backend_info.version,
                'features': backend_info.features
            }
        
        # Module availability
        modules = check_polyglot_availability()
        info['modules'] = modules
        
        # Recommendations
        if not info['backends'].get('rust_core', {}).get('available', False):
            info['recommendations'].append("Install Rust backend for best KV cache performance")
        
        if not info['backends'].get('cpp_core', {}).get('available', False):
            info['recommendations'].append("Install C++ backend for best attention performance")
        
    except Exception as e:
        info['error'] = str(e)
    
    return info


class PolyglotTestHelper:
    """
    Helper class for writing tests with polyglot_core.
    
    Provides utilities for test setup, teardown, and assertions.
    """
    
    def __init__(self):
        self.availability = check_polyglot_availability()
    
    def skip_if_unavailable(self, module: str):
        """Skip test if module is unavailable."""
        import pytest
        
        if not self.availability.get(module, False):
            pytest.skip(f"{module} not available")
    
    def require_backend(self, backend_name: str):
        """Require specific backend for test."""
        import pytest
        
        backend_key = f'backend_{backend_name}'
        if not self.availability.get(backend_key, False):
            pytest.skip(f"Backend {backend_name} not available")
    
    def get_available_backend_for(self, feature: str) -> Optional[str]:
        """
        Get available backend for a feature.
        
        Args:
            feature: Feature name (e.g., 'kv_cache', 'attention')
            
        Returns:
            Backend name or None
        """
        from .backend import get_best_backend, is_backend_available
        
        backend = get_best_backend(feature)
        if is_backend_available(backend):
            return backend.name
        return None
    
    def create_test_cache(self, **kwargs):
        """Create cache for testing."""
        from .cache import KVCache
        return KVCache(max_size=kwargs.get('max_size', 100), **kwargs)
    
    def create_test_attention(self, **kwargs):
        """Create attention for testing."""
        from .attention import Attention, AttentionConfig
        config = AttentionConfig(**kwargs)
        return Attention(config)
    
    def create_test_compressor(self, **kwargs):
        """Create compressor for testing."""
        from .compression import Compressor
        return Compressor(**kwargs)


# Global test helper
_test_helper = PolyglotTestHelper()


def get_test_helper() -> PolyglotTestHelper:
    """Get global test helper instance."""
    return _test_helper













