"""
Adapter layer for abstracting different implementations.
Enables swapping implementations without changing code.
"""
from .model_adapter import ModelAdapter, HuggingFaceModelAdapter
from .data_adapter import DataAdapter, HuggingFaceDataAdapter, JSONLDataAdapter
from .optimizer_adapter import OptimizerAdapter, PyTorchOptimizerAdapter

__all__ = [
    "ModelAdapter",
    "HuggingFaceModelAdapter",
    "DataAdapter",
    "HuggingFaceDataAdapter",
    "JSONLDataAdapter",
    "OptimizerAdapter",
    "PyTorchOptimizerAdapter",
]

