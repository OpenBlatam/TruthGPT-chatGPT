"""
Advanced optimization registry for managing enhanced optimization techniques.
Integrates advanced normalization, positional encodings, enhanced MLP, RL pruning,
enhanced MCTS, and olympiad benchmarking.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import warnings

# Import from techniques sibling package
try:
    from ..techniques import (
        AdvancedNormalizationOptimizations,
        PositionalEncodingOptimizations,
        EnhancedMLPOptimizations,
        RLPruningOptimizations
    )
except ImportError:
    # Fallback for direct execution
    from optimization_core.modules.optimizers.techniques import (
        AdvancedNormalizationOptimizations,
        PositionalEncodingOptimizations,
        EnhancedMLPOptimizations,
        RLPruningOptimizations
    )

@dataclass
class AdvancedOptimizationConfig:
    """Configuration for advanced optimization techniques."""
    enable_advanced_normalization: bool = True
    enable_positional_encodings: bool = True
    enable_enhanced_mlp: bool = True
    enable_rl_pruning: bool = True
    enable_enhanced_mcts: bool = True
    enable_olympiad_benchmarks: bool = True
    
    advanced_normalization_config: Dict[str, Any] = field(default_factory=dict)
    positional_encoding_config: Dict[str, Any] = field(default_factory=dict)
    enhanced_mlp_config: Dict[str, Any] = field(default_factory=dict)
    rl_pruning_config: Dict[str, Any] = field(default_factory=dict)
    enhanced_mcts_config: Dict[str, Any] = field(default_factory=dict)
    olympiad_benchmark_config: Dict[str, Any] = field(default_factory=dict)

# Configs remained unchanged from original
ADVANCED_OPTIMIZATION_CONFIGS = {
    'deepseek_v3': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=False,
        enable_rl_pruning=False,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_llama_rms_norm': True,
            'eps': 1e-6
        },
        positional_encoding_config={
            'use_fixed_llama_rotary_embedding': True,
            'max_seq_len': 4096
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.05,
            'pruning_threshold': 0.005
        },
        olympiad_benchmark_config={
            'problem_categories': ['algebra', 'number_theory'],
            'difficulty_levels': ['amc_12', 'aime'],
            'problems_per_category': 15
        }
    ),
    'qwen': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=False,
        enable_rl_pruning=False,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_advanced_rms_norm': True,
            'eps': 1e-8
        },
        positional_encoding_config={
            'use_rotary_embedding': True,
            'max_seq_len': 2048
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.08
        },
        olympiad_benchmark_config={
            'problem_categories': ['algebra', 'number_theory', 'combinatorics'],
            'difficulty_levels': ['amc_12', 'aime'],
            'problems_per_category': 12
        }
    ),
    'viral_clipper': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=False,
        enable_enhanced_mlp=True,
        enable_rl_pruning=False,
        enable_enhanced_mcts=False,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_crms_norm': True,
            'eps': 1e-8
        },
        enhanced_mlp_config={
            'use_gated_mlp': True
        },
        olympiad_benchmark_config={
            'problem_categories': ['combinatorics', 'geometry'],
            'difficulty_levels': ['amc_12', 'aime'],
            'problems_per_category': 8
        }
    ),
    'brandkit': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=False,
        enable_enhanced_mlp=True,
        enable_rl_pruning=False,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_advanced_rms_norm': True
        },
        enhanced_mlp_config={
            'use_swiglu': True
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.12
        },
        olympiad_benchmark_config={
            'problem_categories': ['algebra', 'geometry'],
            'difficulty_levels': ['amc_12'],
            'problems_per_category': 10
        }
    ),
    'ia_generative': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_advanced_rms_norm': True,
            'use_llama_rms_norm': True,
            'eps': 1e-8
        },
        positional_encoding_config={
            'use_rotary_embedding': True,
            'use_llama_rotary_embedding': True,
            'max_seq_len': 4096
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'use_gated_mlp': True,
            'num_experts': 4,
            'top_k': 2
        },
        rl_pruning_config={
            'target_sparsity': 0.3,
            'use_rl_agent': True
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.1,
            'pruning_threshold': 0.01
        },
        olympiad_benchmark_config={
            'problem_categories': ['algebra', 'number_theory', 'geometry'],
            'difficulty_levels': ['amc_12', 'aime', 'usamo'],
            'problems_per_category': 15
        }
    ),
    'ultra_optimized': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_advanced_rms_norm': True,
            'use_llama_rms_norm': True,
            'use_crms_norm': True,
            'eps': 1e-8
        },
        positional_encoding_config={
            'use_rotary_embedding': True,
            'use_llama_rotary_embedding': True,
            'use_alibi': True,
            'max_seq_len': 8192
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'use_gated_mlp': True,
            'use_mixture_of_experts': True,
            'num_experts': 8,
            'top_k': 2
        },
        rl_pruning_config={
            'target_sparsity': 0.5,
            'use_rl_agent': True,
            'pruning_schedule': 'gradual'
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.1,
            'pruning_threshold': 0.01,
            'policy_temperature': 1.0,
            'neural_guidance_weight': 0.3
        },
        olympiad_benchmark_config={
            'problem_categories': ['algebra', 'number_theory', 'geometry', 'combinatorics'],
            'difficulty_levels': ['amc_12', 'aime', 'usamo', 'imo'],
            'problems_per_category': 20,
            'time_limit_minutes': 60
        }
    ),
    'claude_api': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=False,
        enable_enhanced_mlp=True,
        enable_rl_pruning=False,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_advanced_rms_norm': True,
            'eps': 1e-8
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'use_gated_mlp': True
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.1
        },
        olympiad_benchmark_config={
            'problem_categories': ['algebra', 'number_theory'],
            'difficulty_levels': ['amc_12', 'aime'],
            'problems_per_category': 10
        }
    ),
    'claud_api': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=False,
        enable_enhanced_mlp=True,
        enable_rl_pruning=False,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_advanced_rms_norm': True,
            'eps': 1e-8
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'use_gated_mlp': True
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.1
        },
        olympiad_benchmark_config={
            'problem_categories': ['algebra', 'number_theory'],
            'difficulty_levels': ['amc_12', 'aime'],
            'problems_per_category': 10
        }
    )
}

def get_advanced_optimization_config(variant_name: str) -> AdvancedOptimizationConfig:
    """Get advanced optimization configuration for a specific variant."""
    return ADVANCED_OPTIMIZATION_CONFIGS.get(variant_name, AdvancedOptimizationConfig())

def apply_advanced_optimizations(model: nn.Module, config: AdvancedOptimizationConfig) -> nn.Module:
    """Apply advanced optimizations to a model."""
    optimized_model = model
    
    try:
        if config.enable_advanced_normalization:
            optimized_model = AdvancedNormalizationOptimizations.replace_with_llama_rms_norm(optimized_model)
            print("✅ Applied advanced normalization optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply advanced normalization: {e}")
    
    try:
        if config.enable_positional_encodings:
            optimized_model = PositionalEncodingOptimizations.replace_rotary_embeddings(optimized_model, "fixed_llama")
            print("✅ Applied positional encoding optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply positional encodings: {e}")
    
    try:
        if config.enable_enhanced_mlp:
            optimized_model = EnhancedMLPOptimizations.replace_mlp_with_swiglu(optimized_model)
            print("✅ Applied enhanced MLP optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply enhanced MLP: {e}")
    
    try:
        if config.enable_rl_pruning:
            optimized_model = RLPruningOptimizations.apply_rl_pruning(optimized_model)
            print("✅ Applied RL pruning optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply RL pruning: {e}")
    
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

