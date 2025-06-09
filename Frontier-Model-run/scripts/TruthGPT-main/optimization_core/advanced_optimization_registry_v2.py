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
        enable_rl_pruning=False,  # API models don't need pruning
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
    'claud_api': AdvancedOptimizationConfig(  # Alternative spelling as requested
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
    
    'deepseek_v3_enhanced': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_llama_rms_norm': True,
            'eps': 5e-7,
            'adaptive_eps': True
        },
        positional_encoding_config={
            'use_fixed_llama_rotary_embedding': True,
            'max_seq_len': 8192,
            'rope_scaling_factor': 1.2
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.03,
            'pruning_threshold': 0.003,
            'exploration_temperature': 0.8,
            'value_network_lr': 5e-4
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'intermediate_size_multiplier': 2.7,
            'dropout_rate': 0.05
        }
    ),
    
    'qwen_optimized': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=True,
        advanced_normalization_config={
            'use_llama_rms_norm': True,
            'eps': 8e-7,
            'adaptive_eps': True,
            'momentum_factor': 0.95
        },
        positional_encoding_config={
            'use_fixed_llama_rotary_embedding': True,
            'max_seq_len': 4096,
            'rope_scaling_factor': 1.1,
            'rope_theta': 10000
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.08,
            'pruning_threshold': 0.008,
            'exploration_temperature': 0.9,
            'value_network_lr': 3e-4,
            'policy_network_lr': 1e-4
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'intermediate_size_multiplier': 2.2,
            'dropout_rate': 0.03,
            'activation_scaling': 1.1
        }
    ),
    
    'viral_clipper_optimized': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=False,
        advanced_normalization_config={
            'use_llama_rms_norm': True,
            'eps': 5e-6,
            'adaptive_eps': True,
            'temporal_smoothing': True
        },
        positional_encoding_config={
            'use_temporal_encoding': True,
            'max_temporal_len': 1000,
            'temporal_resolution': 0.1
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.12,
            'pruning_threshold': 0.025,
            'exploration_temperature': 0.7,
            'multimodal_fusion_weight': 0.6
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'intermediate_size_multiplier': 1.8,
            'dropout_rate': 0.08,
            'multimodal_projection': True
        }
    ),
    
    'brandkit_optimized': AdvancedOptimizationConfig(
        enable_advanced_normalization=True,
        enable_positional_encodings=True,
        enable_enhanced_mlp=True,
        enable_rl_pruning=True,
        enable_enhanced_mcts=True,
        enable_olympiad_benchmarks=False,
        advanced_normalization_config={
            'use_llama_rms_norm': True,
            'eps': 3e-6,
            'adaptive_eps': True,
            'brand_consistency_weighting': True
        },
        positional_encoding_config={
            'use_spatial_encoding': True,
            'max_spatial_dim': 512,
            'color_space_encoding': True
        },
        enhanced_mcts_config={
            'use_neural_guidance': True,
            'entropy_weight': 0.15,
            'pruning_threshold': 0.04,
            'exploration_temperature': 0.6,
            'brand_coherence_weight': 0.8
        },
        enhanced_mlp_config={
            'use_swiglu': True,
            'intermediate_size_multiplier': 1.6,
            'dropout_rate': 0.06,
            'brand_feature_projection': True,
            'color_attention_mechanism': True
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
            from .advanced_normalization import AdvancedNormalizationOptimizations
            optimized_model = AdvancedNormalizationOptimizations.replace_with_llama_rms_norm(optimized_model)
            print("✅ Applied advanced normalization optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply advanced normalization: {e}")
    
    try:
        if config.enable_positional_encodings:
            from .positional_encodings import PositionalEncodingOptimizations
            optimized_model = PositionalEncodingOptimizations.replace_rotary_embeddings(optimized_model, "fixed_llama")
            print("✅ Applied positional encoding optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply positional encodings: {e}")
    
    try:
        if config.enable_enhanced_mlp:
            from .enhanced_mlp import EnhancedMLPOptimizations
            optimized_model = EnhancedMLPOptimizations.replace_mlp_with_swiglu(optimized_model)
            print("✅ Applied enhanced MLP optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply enhanced MLP: {e}")
    
    try:
        if config.enable_rl_pruning:
            from .rl_pruning import RLPruningOptimizations
            optimized_model = RLPruningOptimizations.apply_rl_pruning(optimized_model)
            print("✅ Applied RL pruning optimization")
    except Exception as e:
        warnings.warn(f"Failed to apply RL pruning: {e}")
    
    return optimized_model

def get_advanced_optimization_report(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive advanced optimization report."""
    report = {}
    
    try:
        from .advanced_normalization import AdvancedNormalizationOptimizations
        report['normalization'] = AdvancedNormalizationOptimizations.get_normalization_report(model)
    except:
        report['normalization'] = {'error': 'Advanced normalization report not available'}
    
    try:
        from .positional_encodings import PositionalEncodingOptimizations
        report['positional_encodings'] = PositionalEncodingOptimizations.get_positional_encoding_report(model)
    except:
        report['positional_encodings'] = {'error': 'Positional encoding report not available'}
    
    try:
        from .enhanced_mlp import EnhancedMLPOptimizations
        report['enhanced_mlp'] = EnhancedMLPOptimizations.get_mlp_optimization_report(model)
    except:
        report['enhanced_mlp'] = {'error': 'Enhanced MLP report not available'}
    
    try:
        from .rl_pruning import RLPruningOptimizations
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
