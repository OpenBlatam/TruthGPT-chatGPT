"""
Optimization registry for managing and applying optimizations across TruthGPT variants.
Enhanced with MCTS, parallel training, and advanced optimization techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import warnings

@dataclass
class OptimizationConfig:
    """Configuration for optimization techniques."""
    use_cuda_kernels: bool = field(default=True, metadata={"help": "Use CUDA kernels"})
    use_triton_kernels: bool = field(default=True, metadata={"help": "Use Triton kernels"})
    use_enhanced_grpo: bool = field(default=True, metadata={"help": "Use enhanced GRPO training"})
    use_mixed_precision: bool = field(default=True, metadata={"help": "Use mixed precision training"})
    use_gradient_checkpointing: bool = field(default=False, metadata={"help": "Use gradient checkpointing"})
    
    use_mcts_optimization: bool = field(default=True, metadata={"help": "Use MCTS optimization"})
    use_parallel_training: bool = field(default=True, metadata={"help": "Use parallel training"})
    use_experience_buffer: bool = field(default=True, metadata={"help": "Use experience buffer"})
    use_advanced_losses: bool = field(default=True, metadata={"help": "Use advanced loss functions"})
    use_reward_functions: bool = field(default=True, metadata={"help": "Use advanced reward functions"})
    
    cuda_block_size: int = field(default=256, metadata={"help": "CUDA block size"})
    cuda_grid_size: int = field(default=1024, metadata={"help": "CUDA grid size"})
    
    triton_block_size: int = field(default=1024, metadata={"help": "Triton block size"})
    
    grpo_clip_ratio: float = field(default=0.2, metadata={"help": "GRPO clip ratio"})
    grpo_entropy_coef: float = field(default=0.01, metadata={"help": "GRPO entropy coefficient"})
    
    mcts_max_evaluations: int = field(default=100, metadata={"help": "MCTS max evaluations"})
    mcts_exploration_constant: float = field(default=0.1, metadata={"help": "MCTS exploration constant"})
    
    parallel_micro_batch_size: int = field(default=4, metadata={"help": "Parallel training micro batch size"})
    parallel_global_batch_size: int = field(default=8, metadata={"help": "Parallel training global batch size"})

class OptimizationRegistry:
    """Enhanced registry for managing optimization techniques."""
    
    def __init__(self):
        self.optimizations: Dict[str, Callable] = {}
        self.configs: Dict[str, OptimizationConfig] = {}
        self._register_default_optimizations()
    
    def _register_default_optimizations(self):
        """Register default and enhanced optimization techniques."""
        try:
            from .cuda_kernels import CUDAOptimizations
            from .triton_optimizations import apply_triton_optimizations
            from .enhanced_grpo import create_enhanced_grpo_trainer
            try:
                from .mcts_optimization import create_mcts_optimizer
                self.register_optimization("mcts_optimization", create_mcts_optimizer)
            except ImportError:
                warnings.warn("MCTS optimization not available")
            
            try:
                from .parallel_training import create_parallel_actor
                self.register_optimization("parallel_training", create_parallel_actor)
            except ImportError:
                warnings.warn("Parallel training not available")
            
            try:
                from .experience_buffer import create_experience_buffer
                self.register_optimization("experience_buffer", create_experience_buffer)
            except ImportError:
                warnings.warn("Experience buffer not available")
            
            try:
                from .advanced_losses import create_loss_function
                self.register_optimization("advanced_losses", create_loss_function)
            except ImportError:
                warnings.warn("Advanced losses not available")
            
            try:
                from .reward_functions import create_reward_function
                self.register_optimization("reward_functions", create_reward_function)
            except ImportError:
                warnings.warn("Reward functions not available")
            
            self.register_optimization("cuda_kernels", CUDAOptimizations.replace_layer_norm)
            self.register_optimization("triton_kernels", apply_triton_optimizations)
            self.register_optimization("enhanced_grpo", create_enhanced_grpo_trainer)
            self.register_optimization("mcts_optimization", create_mcts_optimizer)
            self.register_optimization("parallel_training", create_parallel_actor)
            self.register_optimization("experience_buffer", create_experience_buffer)
            self.register_optimization("advanced_losses", create_loss_function)
            self.register_optimization("reward_functions", create_reward_function)
        except ImportError as e:
            warnings.warn(f"Some optimizations could not be loaded: {e}")
    
    def register_optimization(self, name: str, optimization_func: Callable):
        """Register a new optimization technique."""
        self.optimizations[name] = optimization_func
        print(f"Registered optimization: {name}")
    
    def apply_optimization(self, model: nn.Module, optimization_name: str, 
                          config: Optional[OptimizationConfig] = None, **kwargs) -> nn.Module:
        """Apply a specific optimization to a model."""
        if optimization_name not in self.optimizations:
            warnings.warn(f"Optimization '{optimization_name}' not found in registry")
            return model
        
        if config is None:
            config = self.configs.get(optimization_name, OptimizationConfig())
        
        try:
            optimization_func = self.optimizations[optimization_name]
            
            if optimization_name == "enhanced_grpo":
                from .enhanced_grpo import EnhancedGRPOArgs
                grpo_args = EnhancedGRPOArgs(
                    clip_ratio=config.grpo_clip_ratio,
                    entropy_coef=config.grpo_entropy_coef,
                    use_mixed_precision=config.use_mixed_precision,
                    gradient_checkpointing=config.use_gradient_checkpointing,
                    use_amp=config.use_mixed_precision,
                    gradient_accumulation_steps=1
                )
                return optimization_func(model, grpo_args)
            
            elif optimization_name == "mcts_optimization":
                from .mcts_optimization import MCTSOptimizationArgs
                
                def dummy_objective(config_dict):
                    return 0.5
                
                mcts_args = MCTSOptimizationArgs(
                    fe_max=config.mcts_max_evaluations,
                    exploration_constant_0=config.mcts_exploration_constant
                )
                return optimization_func(dummy_objective, mcts_args)
            
            elif optimization_name == "parallel_training":
                from .parallel_training import ParallelTrainingConfig
                parallel_config = ParallelTrainingConfig(
                    micro_batch_size_per_device_for_experience=config.parallel_micro_batch_size,
                    global_batch_size_per_device=config.parallel_global_batch_size
                )
                return optimization_func(model, parallel_config)
            
            elif optimization_name == "cuda_kernels":
                return optimization_func(model)
            
            else:
                return optimization_func(model, **kwargs)
                
        except Exception as e:
            warnings.warn(f"Failed to apply optimization '{optimization_name}': {e}")
            return model
    
    def apply_multiple_optimizations(self, model: nn.Module, optimization_names: List[str],
                                   config: Optional[OptimizationConfig] = None) -> nn.Module:
        """Apply multiple optimizations to a model."""
        optimized_model = model
        applied_optimizations = []
        
        for opt_name in optimization_names:
            try:
                optimized_model = self.apply_optimization(optimized_model, opt_name, config)
                applied_optimizations.append(opt_name)
            except Exception as e:
                warnings.warn(f"Skipping optimization '{opt_name}': {e}")
        
        print(f"Applied optimizations: {applied_optimizations}")
        return optimized_model
    
    def get_available_optimizations(self) -> List[str]:
        """Get list of available optimizations."""
        return list(self.optimizations.keys())
    
    def set_config(self, optimization_name: str, config: OptimizationConfig):
        """Set configuration for a specific optimization."""
        self.configs[optimization_name] = config
    
    def get_optimization_report(self, model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        try:
            from .cuda_kernels import CUDAOptimizations
            cuda_report = CUDAOptimizations.get_optimization_report(model)
        except:
            cuda_report = {'cuda_available': False}
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            **cuda_report,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'available_optimizations': self.get_available_optimizations(),
            'registered_configs': list(self.configs.keys())
        }

_optimization_registry = OptimizationRegistry()

def apply_optimizations(model: nn.Module, optimizations: Optional[List[str]] = None,
                       config: Optional[OptimizationConfig] = None) -> nn.Module:
    """Apply optimizations to a model using the global registry."""
    if optimizations is None:
        optimizations = ["cuda_kernels", "enhanced_grpo"]
    
    if isinstance(config, OptimizationConfig):
        return _optimization_registry.apply_multiple_optimizations(model, optimizations, config)
    elif isinstance(config, str):
        actual_config = get_optimization_config(config)
        return _optimization_registry.apply_multiple_optimizations(model, optimizations, actual_config)
    else:
        return _optimization_registry.apply_multiple_optimizations(model, optimizations, config)

def get_optimization_config(variant_name: str = "default") -> OptimizationConfig:
    """Get optimization configuration for a specific variant."""
    configs = {
        "deepseek_v3": OptimizationConfig(
            use_cuda_kernels=True,
            use_triton_kernels=True,
            use_enhanced_grpo=True,
            use_mcts_optimization=True,
            use_parallel_training=True,
            grpo_clip_ratio=0.15,
            grpo_entropy_coef=0.005,
            mcts_max_evaluations=50
        ),
        "qwen_qwq": OptimizationConfig(
            use_cuda_kernels=True,
            use_triton_kernels=True,
            use_enhanced_grpo=True,
            use_parallel_training=True,
            use_experience_buffer=True,
            grpo_clip_ratio=0.18,
            grpo_entropy_coef=0.008
        ),
        "viral_clipper": OptimizationConfig(
            use_cuda_kernels=True,
            use_triton_kernels=False,
            use_enhanced_grpo=True,
            use_experience_buffer=True,
            use_advanced_losses=True,
            grpo_clip_ratio=0.25,
            grpo_entropy_coef=0.02
        ),
        "brandkit": OptimizationConfig(
            use_cuda_kernels=True,
            use_triton_kernels=True,
            use_enhanced_grpo=True,
            use_mixed_precision=True,
            use_mcts_optimization=True,
            use_reward_functions=True,
            grpo_clip_ratio=0.2,
            grpo_entropy_coef=0.01
        ),
        "ia_generative": OptimizationConfig(
            use_cuda_kernels=True,
            use_triton_kernels=True,
            use_enhanced_grpo=True,
            use_parallel_training=True,
            use_advanced_losses=True,
            use_reward_functions=True,
            grpo_clip_ratio=0.22,
            grpo_entropy_coef=0.012
        ),
        "ultra_optimized": OptimizationConfig(
            use_cuda_kernels=True,
            use_triton_kernels=True,
            use_enhanced_grpo=True,
            use_mcts_optimization=True,
            use_parallel_training=True,
            use_experience_buffer=True,
            use_advanced_losses=True,
            use_reward_functions=True,
            grpo_clip_ratio=0.18,
            grpo_entropy_coef=0.008,
            mcts_max_evaluations=75
        ),
        "default": OptimizationConfig()
    }
    
    return configs.get(variant_name, configs["default"])

def register_optimization(name: str, optimization_func: Callable):
    """Register a new optimization in the global registry."""
    _optimization_registry.register_optimization(name, optimization_func)

def get_optimization_report(model: nn.Module) -> Dict[str, Any]:
    """Get optimization report for a model."""
    return _optimization_registry.get_optimization_report(model)
