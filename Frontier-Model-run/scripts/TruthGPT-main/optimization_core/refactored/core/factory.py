"""
Optimizer Factory
================

Factory pattern implementation for creating and managing optimizers.
Supports dynamic optimizer creation, registration, and lifecycle management.
"""

import logging
from typing import Dict, Any, Optional, Type, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import importlib
import inspect

from .container import DependencyContainer


class OptimizerType(Enum):
    """Available optimizer types"""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"
    CUSTOM = "custom"


@dataclass
class OptimizerMetadata:
    """Metadata for optimizer registration"""
    name: str
    type: OptimizerType
    class_path: str
    dependencies: List[str]
    config_schema: Dict[str, Any]
    description: str
    version: str


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers"""
    
    def __init__(self, config: Dict[str, Any], container: DependencyContainer):
        self.config = config
        self.container = container
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = {}
    
    @abstractmethod
    async def optimize(self, data: Any) -> Any:
        """Main optimization method"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate optimizer configuration"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get optimizer metrics"""
        return self.metrics
    
    def cleanup(self):
        """Cleanup resources"""
        pass


class OptimizerFactory:
    """
    Factory for creating and managing optimizers.
    
    Features:
    - Dynamic optimizer registration
    - Dependency injection
    - Configuration validation
    - Lifecycle management
    - Plugin support
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.registered_optimizers: Dict[str, OptimizerMetadata] = {}
        self.optimizer_instances: Dict[str, BaseOptimizer] = {}
        
        # Register built-in optimizers
        self._register_builtin_optimizers()
    
    def _register_builtin_optimizers(self):
        """Register built-in optimizers"""
        builtin_optimizers = [
            {
                'name': 'transformer_optimizer',
                'type': OptimizerType.TRANSFORMER,
                'class_path': 'refactored.optimizers.transformer.TransformerOptimizer',
                'dependencies': ['torch', 'transformers'],
                'config_schema': {
                    'model_name': {'type': 'string', 'required': True},
                    'learning_rate': {'type': 'float', 'default': 1e-4},
                    'batch_size': {'type': 'int', 'default': 32}
                },
                'description': 'Transformer-based optimizer with attention mechanisms',
                'version': '1.0.0'
            },
            {
                'name': 'diffusion_optimizer',
                'type': OptimizerType.DIFFUSION,
                'class_path': 'refactored.optimizers.diffusion.DiffusionOptimizer',
                'dependencies': ['torch', 'diffusers'],
                'config_schema': {
                    'scheduler': {'type': 'string', 'default': 'DDIMScheduler'},
                    'num_inference_steps': {'type': 'int', 'default': 50}
                },
                'description': 'Diffusion model optimizer',
                'version': '1.0.0'
            },
            {
                'name': 'quantum_optimizer',
                'type': OptimizerType.QUANTUM,
                'class_path': 'refactored.optimizers.quantum.QuantumOptimizer',
                'dependencies': ['qiskit', 'numpy'],
                'config_schema': {
                    'quantum_circuit': {'type': 'string', 'required': True},
                    'shots': {'type': 'int', 'default': 1024}
                },
                'description': 'Quantum computing optimizer',
                'version': '1.0.0'
            }
        ]
        
        for optimizer_info in builtin_optimizers:
            metadata = OptimizerMetadata(**optimizer_info)
            self.registered_optimizers[metadata.name] = metadata
    
    def register_optimizer(self, metadata: OptimizerMetadata):
        """Register a new optimizer"""
        self.registered_optimizers[metadata.name] = metadata
        self.logger.info(f"Optimizer registered: {metadata.name}")
    
    def get_available_optimizers(self) -> List[str]:
        """Get list of available optimizer names"""
        return list(self.registered_optimizers.keys())
    
    def get_optimizer_metadata(self, name: str) -> Optional[OptimizerMetadata]:
        """Get metadata for specific optimizer"""
        return self.registered_optimizers.get(name)
    
    async def create_optimizer(self, name: str, config: Dict[str, Any]) -> BaseOptimizer:
        """
        Create optimizer instance with dependency injection
        
        Args:
            name: Optimizer name
            config: Configuration dictionary
            
        Returns:
            Optimizer instance
        """
        # Get metadata
        metadata = self.registered_optimizers.get(name)
        if not metadata:
            raise ValueError(f"Unknown optimizer: {name}")
        
        # Validate configuration
        if not self._validate_config(metadata, config):
            raise ValueError(f"Invalid configuration for optimizer: {name}")
        
        # Check if instance already exists
        instance_key = f"{name}_{hash(str(config))}"
        if instance_key in self.optimizer_instances:
            return self.optimizer_instances[instance_key]
        
        # Import optimizer class
        optimizer_class = self._import_optimizer_class(metadata.class_path)
        
        # Create instance with dependency injection
        optimizer = optimizer_class(config, self.container)
        
        # Store instance
        self.optimizer_instances[instance_key] = optimizer
        
        self.logger.info(f"Created optimizer instance: {name}")
        return optimizer
    
    def _validate_config(self, metadata: OptimizerMetadata, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        schema = metadata.config_schema
        
        for key, requirements in schema.items():
            if requirements.get('required', False) and key not in config:
                self.logger.error(f"Missing required config: {key}")
                return False
            
            if key in config:
                expected_type = requirements.get('type')
                if expected_type and not isinstance(config[key], self._get_type_class(expected_type)):
                    self.logger.error(f"Invalid type for {key}: expected {expected_type}")
                    return False
        
        return True
    
    def _get_type_class(self, type_name: str) -> Type:
        """Get Python type class from string"""
        type_mapping = {
            'string': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        return type_mapping.get(type_name, str)
    
    def _import_optimizer_class(self, class_path: str) -> Type[BaseOptimizer]:
        """Import optimizer class from path"""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import optimizer class {class_path}: {e}")
            raise
    
    def create_optimizer_from_type(self, optimizer_type: OptimizerType, config: Dict[str, Any]) -> BaseOptimizer:
        """Create optimizer by type"""
        # Find optimizer by type
        for name, metadata in self.registered_optimizers.items():
            if metadata.type == optimizer_type:
                return self.create_optimizer(name, config)
        
        raise ValueError(f"No optimizer found for type: {optimizer_type}")
    
    def create_custom_optimizer(self, class_path: str, config: Dict[str, Any]) -> BaseOptimizer:
        """Create custom optimizer from class path"""
        try:
            optimizer_class = self._import_optimizer_class(class_path)
            return optimizer_class(config, self.container)
        except Exception as e:
            self.logger.error(f"Failed to create custom optimizer: {e}")
            raise
    
    def get_optimizer_instance(self, name: str, config: Dict[str, Any]) -> Optional[BaseOptimizer]:
        """Get existing optimizer instance"""
        instance_key = f"{name}_{hash(str(config))}"
        return self.optimizer_instances.get(instance_key)
    
    def cleanup_optimizer(self, name: str, config: Dict[str, Any]):
        """Cleanup optimizer instance"""
        instance_key = f"{name}_{hash(str(config))}"
        if instance_key in self.optimizer_instances:
            optimizer = self.optimizer_instances[instance_key]
            optimizer.cleanup()
            del self.optimizer_instances[instance_key]
            self.logger.info(f"Cleaned up optimizer: {name}")
    
    def cleanup_all_optimizers(self):
        """Cleanup all optimizer instances"""
        for optimizer in self.optimizer_instances.values():
            optimizer.cleanup()
        self.optimizer_instances.clear()
        self.logger.info("Cleaned up all optimizers")
    
    def get_factory_metrics(self) -> Dict[str, Any]:
        """Get factory metrics"""
        return {
            'registered_optimizers': len(self.registered_optimizers),
            'active_instances': len(self.optimizer_instances),
            'available_types': [opt.type.value for opt in self.registered_optimizers.values()]
        }



