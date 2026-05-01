"""
Adaptive Learning Package
========================

Advanced meta-learning and self-evolving optimization capabilities.
"""
from .enums import LearningMode
from .config import AdaptiveLearningConfig
from .tracker import PerformanceTracker
from .meta import MetaLearner
from .engine import SelfImprovementEngine
from .system import AdaptiveLearningSystem

# Factory functions
def create_adaptive_learning_config(**kwargs) -> AdaptiveLearningConfig:
    """Create adaptive learning configuration"""
    return AdaptiveLearningConfig(**kwargs)

def create_adaptive_learning_system(config: AdaptiveLearningConfig) -> AdaptiveLearningSystem:
    """Create adaptive learning system"""
    return AdaptiveLearningSystem(config)

__all__ = [
    'LearningMode',
    'AdaptiveLearningConfig',
    'PerformanceTracker',
    'MetaLearner',
    'SelfImprovementEngine',
    'AdaptiveLearningSystem',
    'create_adaptive_learning_config',
    'create_adaptive_learning_system'
]

