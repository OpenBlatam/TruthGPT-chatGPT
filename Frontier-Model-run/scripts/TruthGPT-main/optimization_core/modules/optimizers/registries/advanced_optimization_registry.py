"""
Advanced optimization registry for managing enhanced optimization techniques.
Integrates advanced normalization, positional encodings, enhanced MLP, and RL pruning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import warnings

# Import from techniques sibling package
try:
    from ..techniques import (
        AdvancedNormalizationOptimizations,
        PositionalEncodingOptimizations,
        EnhancedMLPOptimizations,
        RLPruningOptimizations,
        TritonOptimizations
    )
except ImportError:
    # Fallback for direct execution
    from optimization_core.modules.optimizers.techniques import (
        AdvancedNormalizationOptimizations,
        PositionalEncodingOptimizations,
        EnhancedMLPOptimizations,
        RLPruningOptimizations,
        TritonOptimizations
    )

@dataclass
class AdvancedOptimizationConfig:
    """Configuration for advanced optimization techniques."""
    enable_cuda: bool = False
    enable_triton: bool = False
    enable_enhanced_grpo: bool = False
    enable_mcts: bool = False
    enable_parallel_training: bool = False
    enable_experience_buffer: bool = False
    enable_advanced_losses: bool = False
    enable_reward_functions: bool = False
    enable_advanced_normalization: bool = False
    enable_positional_encodings: bool = False
    enable_enhanced_mlp: bool = False
    enable_rl_pruning: bool = False

def get_advanced_optimizations(config: AdvancedOptimizationConfig) -> List[tuple]:
    """Get list of optimizations to apply based on config."""
    optimizations = []
    
    try:
        if config.enable_cuda:
            # Note: CUDAOptimizations might still be in root or elsewhere
            from optimization_core.utils.cuda_kernels import CUDAOptimizations
            optimizations.append(('cuda', CUDAOptimizations.replace_rms_norm))
    except ImportError:
        warnings.warn("CUDA optimizations not available")
    
    if config.enable_triton:
        optimizations.append(('triton', TritonOptimizations.replace_layer_norm_with_triton))
    
    # Other components like grpo, mcts, etc. are currently imported from parent
    # In a full refactor, these would also move to techniques or other subpackages
    try:
        if config.enable_enhanced_grpo:
            from ..enhanced_grpo import create_enhanced_grpo_trainer
            optimizations.append(('enhanced_grpo', create_enhanced_grpo_trainer))
    except ImportError:
        warnings.warn("Enhanced GRPO not available")
    
    try:
        if config.enable_mcts:
            from ..mcts_optimization import create_mcts_optimizer
            optimizations.append(('mcts', create_mcts_optimizer))
    except ImportError:
        warnings.warn("MCTS optimization not available")
    
    if config.enable_advanced_normalization:
        optimizations.append(('advanced_normalization', AdvancedNormalizationOptimizations.replace_with_llama_rms_norm))
    
    if config.enable_positional_encodings:
        optimizations.append(('positional_encodings', PositionalEncodingOptimizations.replace_rotary_embeddings))
    
    if config.enable_enhanced_mlp:
        optimizations.append(('enhanced_mlp', EnhancedMLPOptimizations.replace_mlp_with_swiglu))
    
    if config.enable_rl_pruning:
        optimizations.append(('rl_pruning', RLPruningOptimizations.apply_rl_pruning))
    
    return optimizations

OPTIMIZATION_CONFIGS = {
    'deepseek_v3': AdvancedOptimizationConfig(
        enable_cuda=True,
        enable_triton=False,
        enable_enhanced_grpo=True,
        enable_mcts=True,
        enable_parallel_training=False,
        enable_experience_buffer=True,
        enable_advanced_losses=True,
        enable_reward_functions=False,
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=False
    ),
    'qwen': AdvancedOptimizationConfig(
        enable_cuda=True,
        enable_triton=False,
        enable_enhanced_grpo=True,
        enable_mcts=False,
        enable_parallel_training=True,
        enable_experience_buffer=True,
        enable_advanced_losses=False,
        enable_reward_functions=False,
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=False,
        enable_rl_pruning=False
    ),
    'viral_clipper': AdvancedOptimizationConfig(
        enable_cuda=True,
        enable_triton=False,
        enable_enhanced_grpo=False,
        enable_mcts=False,
        enable_parallel_training=True,
        enable_experience_buffer=False,
        enable_advanced_losses=True,
        enable_reward_functions=True,
        enable_advanced_normalization=True,
        enable_positional_encodings=False,
        enable_enhanced_mlp=True,
        enable_rl_pruning=False
    ),
    'brandkit': AdvancedOptimizationConfig(
        enable_cuda=False,
        enable_triton=False,
        enable_enhanced_grpo=True,
        enable_mcts=True,
        enable_parallel_training=False,
        enable_experience_buffer=False,
        enable_advanced_losses=False,
        enable_reward_functions=True,
        enable_advanced_normalization=True,
        enable_positional_encodings=False,
        enable_enhanced_mlp=True,
        enable_rl_pruning=False
    ),
    'ia_generative': AdvancedOptimizationConfig(
        enable_cuda=True,
        enable_triton=False,
        enable_enhanced_grpo=True,
        enable_mcts=True,
        enable_parallel_training=True,
        enable_experience_buffer=True,
        enable_advanced_losses=True,
        enable_reward_functions=True,
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True
    ),
    'ultra_optimized': AdvancedOptimizationConfig(
        enable_cuda=True,
        enable_triton=True,
        enable_enhanced_grpo=True,
        enable_mcts=True,
        enable_parallel_training=True,
        enable_experience_buffer=True,
        enable_advanced_losses=True,
        enable_reward_functions=True,
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True
    )
}

def get_advanced_optimization_config(variant_name: str) -> AdvancedOptimizationConfig:
    """Get advanced optimization configuration for a specific variant."""
    return OPTIMIZATION_CONFIGS.get(variant_name, AdvancedOptimizationConfig())

def apply_advanced_optimizations(model: nn.Module, config: AdvancedOptimizationConfig) -> nn.Module:
    """Apply advanced optimizations to a model."""
    optimizations = get_advanced_optimizations(config)
    optimized_model = model
    
    for name, optimization_func in optimizations:
        try:
            if name == 'positional_encodings':
                optimized_model = optimization_func(optimized_model, "fixed_llama")
            else:
                optimized_model = optimization_func(optimized_model)
            print(f"✅ Applied {name} optimization")
        except Exception as e:
            warnings.warn(f"Failed to apply {name} optimization: {e}")
    
    return optimized_model

def get_advanced_optimization_report(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive advanced optimization report."""
    report = {}
    
    try:
        report['normalization'] = AdvancedNormalizationOptimizations.get_normalization_report(model)
    except:
        report['normalization'] = {'error': 'Advanced normalization report not available'}
    
    try:
        report['positional_encodings'] = PositionalEncodingOptimizations.get_positional_encoding_report(model)
    except:
        report['positional_encodings'] = {'error': 'Positional encoding report not available'}
    
    try:
        report['enhanced_mlp'] = EnhancedMLPOptimizations.get_mlp_optimization_report(model)
    except:
        report['enhanced_mlp'] = {'error': 'Enhanced MLP report not available'}
    
    try:
        report['rl_pruning'] = RLPruningOptimizations.get_pruning_report(model)
    except:
        report['rl_pruning'] = {'error': 'RL pruning report not available'}
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    report['model_stats'] = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    return report

