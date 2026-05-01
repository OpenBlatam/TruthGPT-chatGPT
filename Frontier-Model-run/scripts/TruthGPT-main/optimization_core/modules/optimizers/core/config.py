
from optimization_core.core.config import ConfigManager
# Mocking OptimizationConfig if missing, or import if it exists
try:
    from optimization_core.core.config import OptimizationConfig
except ImportError:
    class OptimizationConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def to_dict(self):
            return self.__dict__
        @property
        def enable_quantization(self): return getattr(self, 'quantization', False)
        @property
        def enable_pruning(self): return getattr(self, 'pruning', False)
        @property
        def enable_mixed_precision(self): return getattr(self, 'mixed_precision', False)
        @property
        def enable_kernel_fusion(self): return getattr(self, 'kernel_fusion', False)
        @property
        def level(self): return getattr(self, 'optimization_level', 'standard')

