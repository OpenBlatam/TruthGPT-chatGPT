"""
Multi-Task Learning Package
===========================

Advanced multi-task learning system with shared representations and task balancing.
"""
from .enums import TaskType, TaskRelationship, SharingStrategy
from .config import MultiTaskConfig
from .balancer import TaskBalancer
from .surgery import GradientSurgery
from .layers import SharedRepresentation, MultiTaskHead
from .model import MultiTaskNetwork
from .system import MultiTaskTrainer

# Compatibility aliases
MultitaskModel = MultiTaskNetwork
MultitaskConfig = MultiTaskConfig

# Factory functions
def create_multitask_config(**kwargs) -> MultiTaskConfig:
    return MultiTaskConfig(**kwargs)

def create_task_balancer(config: MultiTaskConfig) -> TaskBalancer:
    return TaskBalancer(config)

def create_gradient_surgery(config: MultiTaskConfig) -> GradientSurgery:
    return GradientSurgery(config)

def create_shared_representation(config: MultiTaskConfig) -> SharedRepresentation:
    return SharedRepresentation(config)

def create_multitask_head(config: MultiTaskConfig, task_type: TaskType) -> MultiTaskHead:
    return MultiTaskHead(config, task_type)

def create_multitask_network(config: MultiTaskConfig) -> MultiTaskNetwork:
    return MultiTaskNetwork(config)

def create_multitask_trainer(config: MultiTaskConfig) -> MultiTaskTrainer:
    return MultiTaskTrainer(config)

__all__ = [
    'TaskType',
    'TaskRelationship',
    'SharingStrategy',
    'MultiTaskConfig',
    'TaskBalancer',
    'GradientSurgery',
    'SharedRepresentation',
    'MultiTaskHead',
    'MultiTaskNetwork',
    'MultiTaskTrainer',
    'create_multitask_config',
    'create_task_balancer',
    'create_gradient_surgery',
    'create_shared_representation',
    'create_multitask_head',
    'create_multitask_network',
    'create_multitask_trainer'
]

