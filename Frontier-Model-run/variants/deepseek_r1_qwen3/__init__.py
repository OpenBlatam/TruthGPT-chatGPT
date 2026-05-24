"""
DeepSeek-R1-Qwen3 Frontier Model

A native implementation combining DeepSeek-R1's advanced reasoning capabilities
with Qwen3's efficient architecture, enhanced with frontier reasoning features.

Key Features:
- Advanced step-by-step reasoning with 23K average tokens per session
- Qwen3 architecture with 8B parameters and optimized attention
- Multi-step verification and confidence calibration
- YARN-scaled RoPE for extended context (131K tokens)
- Chain-of-thought training and curriculum learning
"""

from .model import (
    DeepSeekR1Qwen3Config,
    DeepSeekR1Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3DecoderLayer,
    ReasoningModule,
    RMSNorm,
    RotaryEmbedding,
)

from .trainer import (
    ReasoningTrainer,
    ReasoningTrainingArguments,
    ReasoningDataset,
    ReasoningLossComputer,
    ReasoningEvaluator,
    create_reasoning_data_collator,
    compute_reasoning_metrics,
    setup_reasoning_training,
)

__version__ = "1.0.0"
__author__ = "OpenBlatam Research Team"

__all__ = [
    # Model components
    "DeepSeekR1Qwen3Config",
    "DeepSeekR1Qwen3ForCausalLM",
    "Qwen3Model",
    "Qwen3Attention",
    "Qwen3MLP", 
    "Qwen3DecoderLayer",
    "ReasoningModule",
    "RMSNorm",
    "RotaryEmbedding",
    
    # Training components
    "ReasoningTrainer",
    "ReasoningTrainingArguments",
    "ReasoningDataset",
    "ReasoningLossComputer",
    "ReasoningEvaluator",
    "create_reasoning_data_collator",
    "compute_reasoning_metrics",
    "setup_reasoning_training",
]