"""
Ultra-fast modular training components
Following deep learning best practices.
"""

from __future__ import annotations
from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS = {
    # Training Core
    'FastTrainer': '.trainer',
    'AdvancedTrainer': '.trainer',
    'create_trainer': '.trainer',
    
    # Configuration
    'TrainerConfig': '.config',
    'TrainingConfig': '.config',
    'TrainingStrategy': '.config',
    'OptimizerType': '.config',
    'SchedulerType': '.config',
    'create_training_config': '.config',
    
    # Components
    'TrainingStep': '.components',
    'EMAManager': '.components',
    'CurriculumScheduler': '.components',
    'ExperimentLogger': '.components',
    
    # Data loading
    'FastDataLoader': '.data_loader',
    'DataLoaderConfig': '.data_loader',
    'DataProcessor': '.data_loader',
    
    # Optimization
    'FastOptimizer': '.optimizer',
    'OptimizerConfig': '.optimizer',
    'SchedulerConfig': '.optimizer',
    
    # Loss functions
    'LossFunction': '.loss',
    'LossConfig': '.loss',
    'compute_loss': '.loss',
    
    # Metrics
    'MetricsTracker': '.metrics',
    'MetricConfig': '.metrics',
    'compute_metrics': '.metrics',
    
    # Checkpointing
    'CheckpointManager': '.checkpoint',
    'CheckpointConfig': '.checkpoint',
    'save_checkpoint': '.checkpoint',
    'load_checkpoint': '.checkpoint',
    
    # Validation
    'Validator': '.validation',
    'ValidationConfig': '.validation',
    'validate_model': '.validation',
    
    # Profiling
    'TrainingProfiler': '.profiler',
    'ProfilerConfig': '.profiler',
    'profile_training': '.profiler',
}

def __getattr__(name: str):
    """Lazy import system for training components."""
    return resolve_lazy_import(name, __package__ or 'training', _LAZY_IMPORTS)

__all__ = list(_LAZY_IMPORTS.keys())

