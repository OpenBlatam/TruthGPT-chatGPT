
import logging
from contextlib import contextmanager
from optimization_core.core.performance_utils import (
    measure_memory_usage,
    profile_operation as performance_context
)

class PerformanceUtils:
    def get_inference_time(self, model, input_tensor):
        import time
        start = time.time()
        model(input_tensor)
        return time.time() - start

class MemoryUtils:
    def get_model_memory_usage(self, model):
        # Mocking or simplified implementation
        return {'total_mb': 0.0}
    def get_parameter_count(self, model):
        return {'total_parameters': sum(p.numel() for p in model.parameters())}

class GPUUtils:
    def is_available(self):
        import torch
        return torch.cuda.is_available()

