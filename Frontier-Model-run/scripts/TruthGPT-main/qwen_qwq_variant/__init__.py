"""
Qwen QwQ Variant for TruthGPT
Billion-parameter scale reasoning model with advanced optimizations
"""

from .qwen_qwq_model import QwenQwQForCausalLM, QwenQwQConfig, create_qwen_qwq_model
from .qwen_qwq_optimizations import apply_qwen_qwq_optimizations, create_qwen_qwq_optimizer
from .qwen_qwq_trainer import QwenQwQTrainer, create_qwen_qwq_trainer
from .qwen_qwq_benchmarks import run_qwen_qwq_benchmarks

__all__ = [
    'QwenQwQForCausalLM',
    'QwenQwQConfig', 
    'create_qwen_qwq_model',
    'apply_qwen_qwq_optimizations',
    'create_qwen_qwq_optimizer',
    'QwenQwQTrainer',
    'create_qwen_qwq_trainer',
    'run_qwen_qwq_benchmarks'
]

__version__ = "1.0.0"
