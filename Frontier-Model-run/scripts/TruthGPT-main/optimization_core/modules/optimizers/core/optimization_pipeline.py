"""
Optimization Pipeline
=====================
Chain of Responsibility pattern for applying optimization techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

from optimization_core.modules.optimizers.core.techniques import OptimizationTechnique, get_technique, TechniqueRegistry

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStep:
    """Represents a single optimization step."""
    technique_name: str
    config: Dict[str, Any]
    required: bool = False
    condition: Optional[Callable[[nn.Module], bool]] = None


class OptimizationPipeline:
    """Pipeline for applying multiple optimization techniques in sequence."""
    
    def __init__(self, steps: List[OptimizationStep], registry: TechniqueRegistry = None):
        self.steps = steps
        self.registry = registry or TechniqueRegistry()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.applied_techniques: List[str] = []
        self.failed_techniques: List[str] = []
    
    def apply(self, model: nn.Module) -> Tuple[nn.Module, List[str], List[str]]:
        """
        Apply all optimization steps to the model.
        
        Returns:
            Tuple of (optimized_model, applied_techniques, failed_techniques)
        """
        optimized = model
        
        for step in self.steps:
            if step.condition and not step.condition(optimized):
                self.logger.debug(f"Skipping {step.technique_name} (condition not met)")
                continue
            
            try:
                technique = self.registry.create(step.technique_name, step.config)
                
                if not technique.can_apply(optimized):
                    if step.required:
                        raise RuntimeError(f"Required technique {step.technique_name} cannot be applied")
                    self.logger.warning(f"Technique {step.technique_name} cannot be applied, skipping")
                    continue
                
                optimized, success = technique.apply(optimized)
                
                if success:
                    self.applied_techniques.append(step.technique_name)
                    self.logger.debug(f"Applied {step.technique_name}")
                else:
                    if step.required:
                        raise RuntimeError(f"Required technique {step.technique_name} failed")
                    self.failed_techniques.append(step.technique_name)
                    self.logger.warning(f"Technique {step.technique_name} failed")
            
            except Exception as e:
                if step.required:
                    raise
                self.logger.error(f"Error applying {step.technique_name}: {e}")
                self.failed_techniques.append(step.technique_name)
        
        return optimized, self.applied_techniques, self.failed_techniques


class LevelBasedPipelineBuilder:
    """Builder for creating optimization pipelines based on levels."""
    
    def __init__(self):
        self.level_pipelines: Dict[str, List[OptimizationStep]] = {}
        self._build_default_pipelines()
    
    def _build_default_pipelines(self):
        """Build default pipelines for each optimization level."""
        
        self.level_pipelines['basic'] = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
        ]
        
        self.level_pipelines['advanced'] = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
            OptimizationStep('mixed_precision', {'dtype': 'bf16'}, required=False),
        ]
        
        self.level_pipelines['expert'] = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
            OptimizationStep('mixed_precision', {'dtype': 'bf16'}, required=False),
            OptimizationStep('torch_compile', {'compile_mode': 'default'}, required=False),
        ]
        
        self.level_pipelines['master'] = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
            OptimizationStep('mixed_precision', {'dtype': 'bf16'}, required=False),
            OptimizationStep('torch_compile', {'compile_mode': 'reduce-overhead'}, required=False),
            OptimizationStep('tf32', {}, required=False),
            OptimizationStep('fused_adamw', {}, required=False),
        ]
        
        self.level_pipelines['legendary'] = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
            OptimizationStep('mixed_precision', {'dtype': 'bf16'}, required=False),
            OptimizationStep('torch_compile', {'compile_mode': 'max-autotune'}, required=False),
            OptimizationStep('tf32', {}, required=False),
            OptimizationStep('fused_adamw', {}, required=False),
        ]
        
        self.level_pipelines['transcendent'] = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
            OptimizationStep('mixed_precision', {'dtype': 'bf16'}, required=False),
            OptimizationStep('torch_compile', {'compile_mode': 'max-autotune'}, required=False),
            OptimizationStep('tf32', {}, required=False),
            OptimizationStep('fused_adamw', {}, required=False),
            OptimizationStep('quantization', {'bits': 8}, required=False),
        ]
        
        self.level_pipelines['divine'] = [
            OptimizationStep('gradient_checkpointing', {}, required=False),
            OptimizationStep('mixed_precision', {'dtype': 'bf16'}, required=False),
            OptimizationStep('torch_compile', {'compile_mode': 'max-autotune'}, required=False),
            OptimizationStep('tf32', {}, required=False),
            OptimizationStep('fused_adamw', {}, required=False),
            OptimizationStep('quantization', {'bits': 8}, required=False),
            OptimizationStep('pruning', {'amount': 0.1}, required=False),
        ]
        
        for level in ['omnipotent', 'infinite', 'ultimate', 'supreme', 'enterprise',
                      'ultra_fast', 'ultra_speed', 'super_speed', 'lightning_speed',
                      'hyper_speed', 'extreme']:
            self.level_pipelines[level] = self.level_pipelines['divine'].copy()
    
    def build_pipeline(self, level: str, custom_steps: List[OptimizationStep] = None, config: Dict[str, Any] = None) -> OptimizationPipeline:
        """Build an optimization pipeline for a given level."""
        steps = custom_steps or self.level_pipelines.get(level, self.level_pipelines['basic'])
        
        if config:
            steps = self._apply_config_to_steps(steps, config)
        
        return OptimizationPipeline(steps)
    
    def _apply_config_to_steps(self, steps: List[OptimizationStep], config: Dict[str, Any]) -> List[OptimizationStep]:
        """Apply configuration overrides to pipeline steps."""
        updated_steps = []
        for step in steps:
            step_config = step.config.copy()
            
            if step.technique_name == 'gradient_checkpointing' and 'use_gradient_checkpointing' in config:
                if not config.get('use_gradient_checkpointing', False):
                    continue
            
            if step.technique_name == 'mixed_precision' and 'use_mixed_precision' in config:
                if not config.get('use_mixed_precision', True):
                    continue
                if 'mixed_precision' in config:
                    step_config['dtype'] = config.get('mixed_precision', 'bf16')
            
            if step.technique_name == 'torch_compile' and 'use_torch_compile' in config:
                if not config.get('use_torch_compile', False):
                    continue
                if 'compile_mode' in config:
                    step_config['compile_mode'] = config.get('compile_mode', 'default')
            
            if step.technique_name == 'quantization' and 'use_quantization' in config:
                if not config.get('use_quantization', False):
                    continue
            
            updated_steps.append(OptimizationStep(
                technique_name=step.technique_name,
                config=step_config,
                required=step.required,
                condition=step.condition
            ))
        
        return updated_steps
    
    def add_level(self, level: str, steps: List[OptimizationStep]):
        """Add a custom optimization level."""
        self.level_pipelines[level] = steps


_pipeline_builder = LevelBasedPipelineBuilder()


def build_optimization_pipeline(level: str, custom_steps: List[OptimizationStep] = None, config: Dict[str, Any] = None) -> OptimizationPipeline:
    """Build an optimization pipeline for a given level."""
    return _pipeline_builder.build_pipeline(level, custom_steps, config)


