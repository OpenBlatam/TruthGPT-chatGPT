"""
Optimization Core - Libraries
Modular library system with lazy loading.
"""

from __future__ import annotations
from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS = {
    # Core
    'BaseModule': '.core',
    
    # Models
    'ModelModule': '.models',
    'TransformerModule': '.models',
    'DiffusionModule': '.models',
    'LoRAModule': '.models',
    'QuantizedModule': '.models',
    'create_model_module': '.models',
    
    # Data
    'DataModule': '.data',
    'TextDataModule': '.data',
    'ImageDataModule': '.data',
    'AudioDataModule': '.data',
    'create_data_module': '.data',
    
    # Training
    'TrainingModule': '.training',
    'SupervisedTrainingModule': '.training',
    'UnsupervisedTrainingModule': '.training',
    'create_training_module': '.training',
    
    # Optimization
    'OptimizationModule': '.optimization',
    'AdamWOptimizationModule': '.optimization',
    'LoRAOptimizationModule': '.optimization',
    'create_optimization_module': '.optimization',
    
    # Evaluation
    'EvaluationModule': '.evaluation',
    'ClassificationEvaluationModule': '.evaluation',
    'GenerationEvaluationModule': '.evaluation',
    'create_evaluation_module': '.evaluation',
    
    # Inference
    'InferenceModule': '.inference',
    'TextInferenceModule': '.inference',
    'ImageInferenceModule': '.inference',
    'DiffusionInferenceModule': '.inference',
    'create_inference_module': '.inference',
    
    # Monitoring
    'MonitoringModule': '.monitoring',
    'PerformanceMonitoringModule': '.monitoring',
    'Alert': '.monitoring',
    'create_monitoring_module': '.monitoring',
    
    # Config & System
    'ConfigManager': '.config_manager',
    'ModularSystem': '.system',
}

def __getattr__(name: str):
    """Lazy import system for library components."""
    return resolve_lazy_import(name, __package__ or 'libraries', _LAZY_IMPORTS)

__all__ = list(_LAZY_IMPORTS.keys())

