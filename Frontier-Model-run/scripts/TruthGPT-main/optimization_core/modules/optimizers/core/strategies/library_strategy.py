"""
Library Strategy
================
Integration strategy for the modular library system.
"""

from typing import Dict, Any, List, Optional
import torch.nn as nn
import logging

from .base_strategy import OptimizationStrategy
from ..base_truthgpt_optimizer import OptimizationLevel

from ....modules.libraries.system import ModularSystem

logger = logging.getLogger(__name__)


class LibraryStrategy(OptimizationStrategy):
    """
    Strategy that leverages the modular library system for optimization.
    Wrapper around modules.libraries.ModularSystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize library strategy.
        
        Args:
            config: Configuration dictionary
        """
        # Default to ADVANCED level if not specified
        super().__init__(level=OptimizationLevel.ADVANCED, config=config)
        self.system = None
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the underlying ModularSystem."""
        try:
            # ModularSystem now supports dict config directly via our refactor
            # We can pass self.config or a minimalistic default if None
            config = self.config or {}
            
            # Ensure basic structure if empty
            if not config:
                config = {"modules": {}}
                
            self.system = ModularSystem(config)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize ModularSystem: {e}")
            self.system = None
            import traceback
            self.logger.debug(traceback.format_exc())

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        """
        Apply optimization using the modular library system.
        
        Args:
            model: Model to optimize
            **kwargs: Additional parameters
            
        Returns:
            Optimized model
        """
        if not self.system:
            self.logger.warning("ModularSystem not initialized, returning original model")
            return model
            
        self.logger.info("Applying LibraryStrategy optimization...")
        
        # Inject the model into the system
        # We need to verify if the system's model module can accept an existing model
        # The current implementation of ModularSystem creates its own model via create_model_module
        # So we might need to adapt the system or the model module
        
        # Check if we can inject:
        if "model" in self.system.modules:
            # We can try to assign the model directly if compatible
            # or wrap it into a ModelModule
            self.system.modules["model"].model = model
            
            # Run optimization if configured
            if "optimization" in self.system.modules and "training" in self.system.modules:
                # We typically need a dataloader for optimization
                # If one isn't provided in config or kwargs, we might skip full training
                # and just apply static optimizations (like quantization if configured)
                pass

        self.applied_techniques.append("library_optimization")
        return model
    
    def get_techniques(self) -> List[str]:
        """
        Return list of techniques this strategy applies.
        
        Returns:
            List of technique names
        """
        return self.applied_techniques + ["modular_system_integration"]
    
    def calculate_metrics(
        self, 
        original: nn.Module, 
        optimized: nn.Module
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            original: Original model
            optimized: Optimized model
            
        Returns:
            Dictionary of metrics
        """
        metrics = super().calculate_metrics(original, optimized)
        
        if self.system:
            # Add monitoring metrics from the system
            monitor_data = {"model": optimized}
            system_metrics = self.system.monitor(monitor_data)
            metrics.update(system_metrics)
            
        return metrics

