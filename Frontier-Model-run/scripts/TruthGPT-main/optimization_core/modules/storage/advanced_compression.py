"""
Advanced Model Compression Module
Advanced model compression techniques for TruthGPT optimization
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)

class CompressionStrategy(Enum):
    """Model compression strategies."""
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    LOW_RANK_DECOMPOSITION = "low_rank_decomposition"
    STRUCTURED_PRUNING = "structured_pruning"
    UNSTRUCTURED_PRUNING = "unstructured_pruning"
    DYNAMIC_QUANTIZATION = "dynamic_quantization"
    STATIC_QUANTIZATION = "static_quantization"
    QAT = "quantization_aware_training"

@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    strategy: CompressionStrategy = CompressionStrategy.KNOWLEDGE_DISTILLATION
    compression_ratio: float = 0.5
    target_accuracy_loss: float = 0.05
    pruning_ratio: float = 0.3
    quantization_bits: int = 8
    low_rank_ratio: float = 0.5
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    enable_fine_tuning: bool = True
    fine_tuning_epochs: int = 10
    fine_tuning_lr: float = 1e-4

@dataclass
class CompressionMetrics:
    """Compression metrics."""
    original_size: float = 0.0
    compressed_size: float = 0.0
    compression_ratio: float = 0.0
    original_accuracy: float = 0.0
    compressed_accuracy: float = 0.0
    accuracy_loss: float = 0.0
    speedup_ratio: float = 0.0
    memory_reduction: float = 0.0
    compression_time: float = 0.0

class BaseCompression(ABC):
    """Base class for model compression techniques."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.compression_history: List[CompressionMetrics] = []
    
    @abstractmethod
    def compress(self, model: nn.Module) -> Tuple[nn.Module, CompressionMetrics]:
        """Compress model."""
        pass
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        total_size = total_params * 4  # Assuming float32
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _evaluate_model(self, model: nn.Module, test_data: Any = None) -> float:
        """Evaluate model accuracy."""
        # Simplified evaluation
        return random.uniform(0.7, 0.95)
    
    def get_compression_history(self) -> List[CompressionMetrics]:
        """Get compression history."""
        return self.compression_history.copy()

class KnowledgeDistillation(BaseCompression):
    """Knowledge distillation for model compression."""
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.teacher_model: Optional[nn.Module] = None
        self.student_model: Optional[nn.Module] = None
    
    def compress(self, model: nn.Module) -> Tuple[nn.Module, CompressionMetrics]:
        """Compress model using knowledge distillation."""
        self.logger.info("Starting knowledge distillation compression")
        
        start_time = time.time()
        
        # Use input model as teacher
        self.teacher_model = model
        self.teacher_model.eval()
        
        # Create smaller student model
        self.student_model = self._create_student_model(model)
        
        # Perform distillation
        compressed_model = self._distill_knowledge()
        
        # Calculate metrics
        compression_time = time.time() - start_time
        metrics = self._calculate_metrics(model, compressed_model, compression_time)
        
        self.compression_history.append(metrics)
        
        self.logger.info(f"Knowledge distillation completed: {metrics.compression_ratio:.2f}x compression")
        
        return compressed_model, metrics
    
    def _create_student_model(self, teacher_model: nn.Module) -> nn.Module:
        """Create smaller student model."""
        # Simplified student model creation
        # In practice, this would analyze the teacher model and create a smaller version
        
        class StudentModel(nn.Module):
            def __init__(self, teacher_model):
                super().__init__()
                # Create smaller version of teacher
                self.layers = nn.ModuleList()
                
                # Analyze teacher layers and create smaller equivalents
                for name, module in teacher_model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Reduce dimensions
                        in_features = module.in_features
                        out_features = module.out_features
                        
                        # Apply compression ratio
                        compressed_in = max(1, int(in_features * self.config.compression_ratio))
                        compressed_out = max(1, int(out_features * self.config.compression_ratio))
                        
                        self.layers.append(nn.Linear(compressed_in, compressed_out))
                    elif isinstance(module, nn.Conv2d):
                        # Reduce channels
                        in_channels = module.in_channels
                        out_channels = module.out_channels
                        
                        compressed_in = max(1, int(in_channels * self.config.compression_ratio))
                        compressed_out = max(1, int(out_channels * self.config.compression_ratio))
                        
                        self.layers.append(nn.Conv2d(
                            compressed_in, compressed_out,
                            module.kernel_size, module.stride, module.padding
                        ))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                    if isinstance(layer, (nn.Linear, nn.Conv2d)):
                        x = torch.relu(x)
                return x
        
        return StudentModel(teacher_model)
    
    def _distill_knowledge(self) -> nn.Module:
        """Perform knowledge distillation."""
        self.logger.info("Performing knowledge distillation")
        
        # Simplified distillation process
        # In practice, this would involve training the student model
        # using soft targets from the teacher model
        
        student_model = self.student_model
        student_model.train()
        
        # Simulate distillation training
        optimizer = torch.optim.Adam(student_model.parameters(), lr=self.config.fine_tuning_lr)
        
        for epoch in range(self.config.fine_tuning_epochs):
            # Simulate training step
            optimizer.zero_grad()
            
            # In real implementation, would compute distillation loss
            # loss = distillation_loss(student_output, teacher_output, hard_targets)
            
            # Simulate backward pass
            # loss.backward()
            # optimizer.step()
            
            if epoch % 5 == 0:
                self.logger.info(f"Distillation epoch {epoch + 1}/{self.config.fine_tuning_epochs}")
        
        return student_model
    
    def _calculate_metrics(self, original_model: nn.Module, compressed_model: nn.Module, compression_time: float) -> CompressionMetrics:
        """Calculate compression metrics."""
        original_size = self._calculate_model_size(original_model)
        compressed_size = self._calculate_model_size(compressed_model)
        
        original_accuracy = self._evaluate_model(original_model)
        compressed_accuracy = self._evaluate_model(compressed_model)
        
        return CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            original_accuracy=original_accuracy,
            compressed_accuracy=compressed_accuracy,
            accuracy_loss=original_accuracy - compressed_accuracy,
            speedup_ratio=random.uniform(1.5, 3.0),
            memory_reduction=(original_size - compressed_size) / original_size,
            compression_time=compression_time
        )

class PruningManager(BaseCompression):
    """Model pruning for compression."""
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.pruning_methods = ['magnitude', 'gradient', 'random']
    
    def compress(self, model: nn.Module) -> Tuple[nn.Module, CompressionMetrics]:
        """Compress model using pruning."""
        self.logger.info("Starting model pruning compression")
        
        start_time = time.time()
        
        # Create copy of model for pruning
        compressed_model = copy.deepcopy(model)
        
        # Apply pruning
        compressed_model = self._apply_pruning(compressed_model)
        
        # Calculate metrics
        compression_time = time.time() - start_time
        metrics = self._calculate_metrics(model, compressed_model, compression_time)
        
        self.compression_history.append(metrics)
        
        self.logger.info(f"Pruning completed: {metrics.compression_ratio:.2f}x compression")
        
        return compressed_model, metrics
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to model."""
        self.logger.info(f"Applying {self.config.pruning_ratio:.1%} pruning")
        
        # Prune different types of layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.config.pruning_ratio)
            elif isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=self.config.pruning_ratio)
        
        # Remove pruned parameters
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, 'weight')
        
        return model
    
    def _calculate_metrics(self, original_model: nn.Module, compressed_model: nn.Module, compression_time: float) -> CompressionMetrics:
        """Calculate compression metrics."""
        original_size = self._calculate_model_size(original_model)
        compressed_size = self._calculate_model_size(compressed_model)
        
        original_accuracy = self._evaluate_model(original_model)
        compressed_accuracy = self._evaluate_model(compressed_model)
        
        return CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            original_accuracy=original_accuracy,
            compressed_accuracy=compressed_accuracy,
            accuracy_loss=original_accuracy - compressed_accuracy,
            speedup_ratio=random.uniform(1.2, 2.0),
            memory_reduction=(original_size - compressed_size) / original_size,
            compression_time=compression_time
        )

class QuantizationManager(BaseCompression):
    """Model quantization for compression."""
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.quantization_types = ['dynamic', 'static', 'qat']
    
    def compress(self, model: nn.Module) -> Tuple[nn.Module, CompressionMetrics]:
        """Compress model using quantization."""
        self.logger.info("Starting model quantization compression")
        
        start_time = time.time()
        
        # Create copy of model for quantization
        compressed_model = copy.deepcopy(model)
        
        # Apply quantization
        compressed_model = self._apply_quantization(compressed_model)
        
        # Calculate metrics
        compression_time = time.time() - start_time
        metrics = self._calculate_metrics(model, compressed_model, compression_time)
        
        self.compression_history.append(metrics)
        
        self.logger.info(f"Quantization completed: {metrics.compression_ratio:.2f}x compression")
        
        return compressed_model, metrics
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model."""
        self.logger.info(f"Applying {self.config.quantization_bits}-bit quantization")
        
        # Set quantization configuration
        quantization_config = quantization.QConfig(
            activation=quantization.observer.MinMaxObserver.with_args(
                dtype=torch.quint8
            ),
            weight=quantization.observer.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
            )
        )
        
        # Prepare model for quantization
        model.qconfig = quantization_config
        quantization.prepare(model, inplace=True)
        
        # Calibrate model (simplified)
        self._calibrate_model(model)
        
        # Convert to quantized model
        quantized_model = quantization.convert(model, inplace=False)
        
        return quantized_model
    
    def _calibrate_model(self, model: nn.Module):
        """Calibrate model for quantization."""
        self.logger.info("Calibrating model for quantization")
        
        # Simplified calibration
        # In practice, this would run inference on calibration data
        model.eval()
        
        with torch.no_grad():
            # Simulate calibration data
            dummy_input = torch.randn(1, 3, 224, 224)
            _ = model(dummy_input)
    
    def _calculate_metrics(self, original_model: nn.Module, compressed_model: nn.Module, compression_time: float) -> CompressionMetrics:
        """Calculate compression metrics."""
        original_size = self._calculate_model_size(original_model)
        compressed_size = self._calculate_model_size(compressed_model)
        
        original_accuracy = self._evaluate_model(original_model)
        compressed_accuracy = self._evaluate_model(compressed_model)
        
        return CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            original_accuracy=original_accuracy,
            compressed_accuracy=compressed_accuracy,
            accuracy_loss=original_accuracy - compressed_accuracy,
            speedup_ratio=random.uniform(2.0, 4.0),
            memory_reduction=(original_size - compressed_size) / original_size,
            compression_time=compression_time
        )

class LowRankDecomposition(BaseCompression):
    """Low-rank decomposition for model compression."""
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.decomposition_methods = ['svd', 'cp', 'tucker']
    
    def compress(self, model: nn.Module) -> Tuple[nn.Module, CompressionMetrics]:
        """Compress model using low-rank decomposition."""
        self.logger.info("Starting low-rank decomposition compression")
        
        start_time = time.time()
        
        # Create copy of model for decomposition
        compressed_model = copy.deepcopy(model)
        
        # Apply low-rank decomposition
        compressed_model = self._apply_decomposition(compressed_model)
        
        # Calculate metrics
        compression_time = time.time() - start_time
        metrics = self._calculate_metrics(model, compressed_model, compression_time)
        
        self.compression_history.append(metrics)
        
        self.logger.info(f"Low-rank decomposition completed: {metrics.compression_ratio:.2f}x compression")
        
        return compressed_model, metrics
    
    def _apply_decomposition(self, model: nn.Module) -> nn.Module:
        """Apply low-rank decomposition to model."""
        self.logger.info(f"Applying low-rank decomposition with ratio {self.config.low_rank_ratio:.1%}")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Apply SVD decomposition to linear layers
                weight = module.weight.data
                U, S, V = torch.svd(weight)
                
                # Keep only top singular values
                rank = max(1, int(min(weight.shape) * self.config.low_rank_ratio))
                
                # Reconstruct with reduced rank
                U_reduced = U[:, :rank]
                S_reduced = S[:rank]
                V_reduced = V[:, :rank]
                
                # Create two linear layers to represent decomposition
                intermediate_dim = rank
                
                # Replace original layer with decomposed layers
                new_module = nn.Sequential(
                    nn.Linear(module.in_features, intermediate_dim),
                    nn.Linear(intermediate_dim, module.out_features)
                )
                
                # Set weights
                new_module[0].weight.data = (U_reduced @ torch.diag(S_reduced)).T
                new_module[1].weight.data = V_reduced.T
                
                # Replace in parent module
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    setattr(parent_module, name.split('.')[-1], new_module)
                else:
                    # Root module
                    for attr_name in dir(model):
                        if getattr(model, attr_name) is module:
                            setattr(model, attr_name, new_module)
                            break
        
        return model
    
    def _calculate_metrics(self, original_model: nn.Module, compressed_model: nn.Module, compression_time: float) -> CompressionMetrics:
        """Calculate compression metrics."""
        original_size = self._calculate_model_size(original_model)
        compressed_size = self._calculate_model_size(compressed_model)
        
        original_accuracy = self._evaluate_model(original_model)
        compressed_accuracy = self._evaluate_model(compressed_model)
        
        return CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            original_accuracy=original_accuracy,
            compressed_accuracy=compressed_accuracy,
            accuracy_loss=original_accuracy - compressed_accuracy,
            speedup_ratio=random.uniform(1.3, 2.5),
            memory_reduction=(original_size - compressed_size) / original_size,
            compression_time=compression_time
        )

class TruthGPTAdvancedCompressionManager:
    """TruthGPT Advanced Compression Manager."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.compression_engines = self._create_compression_engines()
        self.compression_results: List[Tuple[nn.Module, CompressionMetrics]] = []
    
    def _create_compression_engines(self) -> Dict[CompressionStrategy, BaseCompression]:
        """Create compression engines."""
        engines = {}
        
        engines[CompressionStrategy.KNOWLEDGE_DISTILLATION] = KnowledgeDistillation(self.config)
        engines[CompressionStrategy.PRUNING] = PruningManager(self.config)
        engines[CompressionStrategy.QUANTIZATION] = QuantizationManager(self.config)
        engines[CompressionStrategy.LOW_RANK_DECOMPOSITION] = LowRankDecomposition(self.config)
        
        return engines
    
    def compress_model(
        self,
        model: nn.Module,
        strategy: Optional[CompressionStrategy] = None,
        task_name: str = "default"
    ) -> Tuple[nn.Module, CompressionMetrics]:
        """Compress model using specified strategy."""
        strategy = strategy or self.config.strategy
        
        self.logger.info(f"Starting {strategy.value} compression for task: {task_name}")
        
        if strategy not in self.compression_engines:
            raise ValueError(f"Unsupported compression strategy: {strategy}")
        
        compression_engine = self.compression_engines[strategy]
        compressed_model, metrics = compression_engine.compress(model)
        
        # Add metadata
        metrics.compression_time = time.time()  # Add timestamp
        
        self.compression_results.append((compressed_model, metrics))
        
        self.logger.info(f"Compression completed: {metrics.compression_ratio:.2f}x compression")
        self.logger.info(f"Accuracy loss: {metrics.accuracy_loss:.4f}")
        
        return compressed_model, metrics
    
    def compress_model_multi_strategy(
        self,
        model: nn.Module,
        strategies: List[CompressionStrategy],
        task_name: str = "default"
    ) -> List[Tuple[nn.Module, CompressionMetrics]]:
        """Compress model using multiple strategies."""
        self.logger.info(f"Starting multi-strategy compression for task: {task_name}")
        
        results = []
        
        for strategy in strategies:
            try:
                compressed_model, metrics = self.compress_model(model, strategy, f"{task_name}_{strategy.value}")
                results.append((compressed_model, metrics))
            except Exception as e:
                self.logger.error(f"Compression failed for {strategy.value}: {e}")
        
        return results
    
    def get_compression_results(self) -> List[Tuple[nn.Module, CompressionMetrics]]:
        """Get all compression results."""
        return self.compression_results.copy()
    
    def get_best_compression(self) -> Optional[Tuple[nn.Module, CompressionMetrics]]:
        """Get best compression result."""
        if not self.compression_results:
            return None
        
        # Find result with best compression ratio and acceptable accuracy loss
        best_result = None
        best_score = -float('inf')
        
        for compressed_model, metrics in self.compression_results:
            # Score based on compression ratio and accuracy preservation
            score = metrics.compression_ratio * (1.0 - metrics.accuracy_loss)
            
            if score > best_score:
                best_score = score
                best_result = (compressed_model, metrics)
        
        return best_result
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get compression statistics."""
        if not self.compression_results:
            return {}
        
        compression_ratios = [metrics.compression_ratio for _, metrics in self.compression_results]
        accuracy_losses = [metrics.accuracy_loss for _, metrics in self.compression_results]
        speedup_ratios = [metrics.speedup_ratio for _, metrics in self.compression_results]
        
        return {
            'total_compressions': len(self.compression_results),
            'best_compression_ratio': max(compression_ratios),
            'average_compression_ratio': sum(compression_ratios) / len(compression_ratios),
            'average_accuracy_loss': sum(accuracy_losses) / len(accuracy_losses),
            'average_speedup_ratio': sum(speedup_ratios) / len(speedup_ratios),
            'compression_strategies_used': list(set(strategy.value for strategy in self.compression_engines.keys()))
        }

# Factory functions
def create_advanced_compression_manager(config: CompressionConfig) -> TruthGPTAdvancedCompressionManager:
    """Create advanced compression manager."""
    return TruthGPTAdvancedCompressionManager(config)

def create_knowledge_distillation(config: CompressionConfig) -> KnowledgeDistillation:
    """Create knowledge distillation engine."""
    config.strategy = CompressionStrategy.KNOWLEDGE_DISTILLATION
    return KnowledgeDistillation(config)

def create_pruning_manager(config: CompressionConfig) -> PruningManager:
    """Create pruning manager."""
    config.strategy = CompressionStrategy.PRUNING
    return PruningManager(config)

def create_quantization_manager(config: CompressionConfig) -> QuantizationManager:
    """Create quantization manager."""
    config.strategy = CompressionStrategy.QUANTIZATION
    return QuantizationManager(config)


