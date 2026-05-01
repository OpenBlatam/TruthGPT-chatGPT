"""
Model Compression for TruthGPT
Following deep learning best practices for model optimization
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class CompressionConfig:
    """Model compression configuration"""
    compression_ratio: float = 0.5  # Target compression ratio
    pruning_ratio: float = 0.3      # Fraction of weights to prune
    quantization_bits: int = 8      # Quantization bits
    use_dynamic_quantization: bool = True
    use_static_quantization: bool = False
    use_knowledge_distillation: bool = True
    teacher_temperature: float = 3.0
    student_temperature: float = 1.0


class ModelCompressor:
    """Main model compression coordinator"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.compression_stats = {}
    
    def compress_model(self, model: nn.Module, 
                      teacher_model: Optional[nn.Module] = None) -> nn.Module:
        """Apply comprehensive model compression"""
        compressed_model = model
        
        # 1. Pruning
        if self.config.pruning_ratio > 0:
            compressed_model = self._apply_pruning(compressed_model)
        
        # 2. Quantization
        if self.config.quantization_bits < 32:
            compressed_model = self._apply_quantization(compressed_model)
        
        # 3. Knowledge Distillation
        if self.config.use_knowledge_distillation and teacher_model is not None:
            compressed_model = self._apply_knowledge_distillation(
                compressed_model, teacher_model
            )
        
        return compressed_model
    
    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured and unstructured pruning"""
        pruned_model = model
        
        # Unstructured pruning
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Apply L1 unstructured pruning
                prune.l1_unstructured(
                    module, 
                    name='weight', 
                    amount=self.config.pruning_ratio
                )
        
        # Make pruning permanent
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, 'weight')
        
        return pruned_model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model"""
        if self.config.use_dynamic_quantization:
            return torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear, nn.LSTM, nn.GRU}, 
                dtype=torch.qint8
            )
        elif self.config.use_static_quantization:
            # Prepare model for static quantization
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model = torch.quantization.prepare(model)
            # Calibration would happen here
            model = torch.quantization.convert(model)
            return model
        else:
            return model
    
    def _apply_knowledge_distillation(self, student_model: nn.Module, 
                                    teacher_model: nn.Module) -> nn.Module:
        """Apply knowledge distillation"""
        distillation = KnowledgeDistillation(
            teacher_model, 
            student_model,
            temperature_teacher=self.config.teacher_temperature,
            temperature_student=self.config.student_temperature
        )
        
        return distillation.get_distilled_model()


class PruningOptimizer:
    """Advanced pruning optimization"""
    
    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio
        self.pruning_masks = {}
    
    def magnitude_pruning(self, model: nn.Module) -> nn.Module:
        """Apply magnitude-based pruning"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Calculate magnitude threshold
                weights = module.weight.data
                threshold = torch.quantile(torch.abs(weights), self.pruning_ratio)
                
                # Create mask
                mask = torch.abs(weights) > threshold
                self.pruning_masks[name] = mask
                
                # Apply mask
                module.weight.data *= mask.float()
        
        return model
    
    def gradient_based_pruning(self, model: nn.Module, 
                             gradients: Dict[str, torch.Tensor]) -> nn.Module:
        """Apply gradient-based pruning"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in gradients:
                # Calculate gradient magnitude
                grad_magnitude = torch.abs(gradients[name])
                threshold = torch.quantile(grad_magnitude, self.pruning_ratio)
                
                # Create mask based on gradient magnitude
                mask = grad_magnitude > threshold
                self.pruning_masks[name] = mask
                
                # Apply mask
                module.weight.data *= mask.float()
        
        return model
    
    def lottery_ticket_pruning(self, model: nn.Module, 
                             initial_weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Apply lottery ticket hypothesis pruning"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in initial_weights:
                # Calculate weight importance
                current_weights = module.weight.data
                initial_weights_tensor = initial_weights[name]
                
                # Calculate importance as difference from initial
                importance = torch.abs(current_weights - initial_weights_tensor)
                threshold = torch.quantile(importance, self.pruning_ratio)
                
                # Create mask
                mask = importance > threshold
                self.pruning_masks[name] = mask
                
                # Apply mask
                module.weight.data *= mask.float()
        
        return model


class KnowledgeDistillation:
    """Knowledge distillation for model compression"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature_teacher: float = 3.0, temperature_student: float = 1.0,
                 alpha: float = 0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature_teacher = temperature_teacher
        self.temperature_student = temperature_student
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def distillation_loss(self, student_outputs: torch.Tensor, 
                          teacher_outputs: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
        """Calculate knowledge distillation loss"""
        # Soft targets from teacher
        teacher_soft = F.softmax(teacher_outputs / self.temperature_teacher, dim=-1)
        student_soft = F.log_softmax(student_outputs / self.temperature_student, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = F.kl_div(
            student_soft, teacher_soft, reduction='batchmean'
        ) * (self.temperature_student ** 2)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def get_distilled_model(self) -> nn.Module:
        """Get the distilled student model"""
        return self.student_model
    
    def train_distillation(self, data_loader, optimizer: torch.optim.Optimizer,
                          num_epochs: int = 10) -> nn.Module:
        """Train with knowledge distillation"""
        self.student_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Get teacher outputs (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)
                
                # Get student outputs
                student_outputs = self.student_model(**batch)
                
                # Calculate distillation loss
                loss = self.distillation_loss(
                    student_outputs, teacher_outputs, batch['labels']
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}")
        
        return self.student_model


class QuantizationOptimizer:
    """Advanced quantization optimization"""
    
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits
    
    def dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization"""
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
    
    def static_quantization(self, model: nn.Module, 
                           calibration_data: List[torch.Tensor]) -> nn.Module:
        """Apply static quantization with calibration"""
        # Set quantization config
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        prepared_model = torch.quantization.prepare(model)
        
        # Calibration
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model
    
    def qat_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization-aware training"""
        # Set QAT config
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        qat_model = torch.quantization.prepare_qat(model)
        
        return qat_model


class CompressionAnalyzer:
    """Analyze compression effectiveness"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_compression(self, original_model: nn.Module, 
                          compressed_model: nn.Module) -> Dict[str, Any]:
        """Analyze compression results"""
        # Model size analysis
        original_size = self._calculate_model_size(original_model)
        compressed_size = self._calculate_model_size(compressed_model)
        compression_ratio = compressed_size / original_size
        
        # Parameter count analysis
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        param_reduction = (original_params - compressed_params) / original_params
        
        # Memory footprint analysis
        original_memory = self._calculate_memory_usage(original_model)
        compressed_memory = self._calculate_memory_usage(compressed_model)
        memory_reduction = (original_memory - compressed_memory) / original_memory
        
        analysis = {
            'original_size_mb': original_size,
            'compressed_size_mb': compressed_size,
            'compression_ratio': compression_ratio,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'param_reduction': param_reduction,
            'original_memory_mb': original_memory,
            'compressed_memory_mb': compressed_memory,
            'memory_reduction': memory_reduction
        }
        
        self.analysis_results = analysis
        return analysis
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _calculate_memory_usage(self, model: nn.Module) -> float:
        """Calculate memory usage in MB"""
        model_size = self._calculate_model_size(model)
        # Add buffer for activations (rough estimate)
        buffer_size = model_size * 0.5
        return model_size + buffer_size
    
    def get_compression_efficiency(self) -> Dict[str, float]:
        """Get compression efficiency metrics"""
        if not self.analysis_results:
            return {}
        
        return {
            'compression_efficiency': 1.0 - self.analysis_results['compression_ratio'],
            'parameter_efficiency': self.analysis_results['param_reduction'],
            'memory_efficiency': self.analysis_results['memory_reduction']
        }



