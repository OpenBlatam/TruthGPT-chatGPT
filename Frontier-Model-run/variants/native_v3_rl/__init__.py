"""
Native DeepSeek-V3 with Reinforcement Learning

A native implementation of the DeepSeek-V3 architecture enhanced with
advanced reinforcement learning capabilities.

Key Features:
- Multi-Head Latent Attention (MLA) with LoRA-style compression
- Mixture of Experts with routed and shared experts
- Advanced RoPE with YARN scaling
- Multi-objective reinforcement learning
- PPO-based policy optimization
- Curiosity-driven exploration
- Experience replay
"""

from .model import (
    NativeV3RLConfig,
    NativeV3RLForCausalLM,
    NativeV3RLModel,
    MultiHeadLatentAttention,
    MoELayer,
    CuriosityModule,
    ValueHead,
    RMSNorm,
    RotaryEmbedding
)

from .trainer import (
    NativeV3RLTrainer,
    RewardComputer,
    RLTrainingState
)

__version__ = "1.0.0"
__author__ = "OpenBlatam Research Team"

__all__ = [
    # Model components
    "NativeV3RLConfig",
    "NativeV3RLForCausalLM", 
    "NativeV3RLModel",
    "MultiHeadLatentAttention",
    "MoELayer",
    "CuriosityModule",
    "ValueHead",
    "RMSNorm",
    "RotaryEmbedding",
    
    # Training components
    "NativeV3RLTrainer",
    "RewardComputer", 
    "RLTrainingState",
]