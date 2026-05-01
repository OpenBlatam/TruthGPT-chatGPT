"""
Checkpointing shim for modules.base.
"""
from ..training.legacy_trainers.checkpoint_manager import (
    CheckpointManager,
)

class CheckpointConfig:
    """Mock config for backward compatibility."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def save_checkpoint(*args, **kwargs):
    pass

def load_checkpoint(*args, **kwargs):
    pass

__all__ = ['CheckpointManager', 'CheckpointConfig', 'save_checkpoint', 'load_checkpoint']
