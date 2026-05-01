"""
TruthGPT Model Compression Module
Advanced model compression techniques for TruthGPT models
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TruthGPTCompressionConfig:
    """Configuration for TruthGPT model compression."""
    # Compression types
    enable_quantization: bool = False
    enable_pruning: bool = False
    enable_distillation: bool = False
    enable_low_rank: bool = False
    
    # Quantization settings
    quantization_type: str = "dynamic"  # dynamic, static, qat
    quantization_bits: int = 8
    quantization_scheme: str = "symmetric"  # symmetric, asymmetric
    
    # Pruning settings
    pruning_type: str = "unstructured"  # unstructured, structured, global
    pruning_ratio: float = 0.1
    pruning_schedule: str = "one_shot"  # one_shot, gradual, lottery_ticket
    
    # Distillation settings
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5
    
    # Low-rank settings
    rank_ratio: float = 0.5
    enable_adaptive_rank: bool = False
    
    # Performance settings
    enable_compression_analysis: bool = True
    target_compression_ratio: float = 0.5
    max_accuracy_loss: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_quantization': self.enable_quantization,
            'enable_pruning': self.enable_pruning,
            'enable_distillation': self.enable_distillation,
            'enable_low_rank': self.enable_low_rank,
            'quantization_type': self.quantization_type,
            'quantization_bits': self.quantization_bits,
            'quantization_scheme': self.quantization_scheme,
            'pruning_type': self.pruning_type,
            'pruning_ratio': self.pruning_ratio,
            'pruning_schedule': self.pruning_schedule,
            'teacher_model_path': self.teacher_model_path,
            'distillation_temperature': self.distillation_temperature,
            'distillation_alpha': self.distillation_alpha,
            'rank_ratio': self.rank_ratio,
            'enable_adaptive_rank': self.enable_adaptive_rank,
            'enable_compression_analysis': self.enable_compression_analysis,
            'target_compression_ratio': self.target_compression_ratio,
            'max_accuracy_loss': self.max_accuracy_loss
        }

class TruthGPTQuantizer:
    """Advanced quantizer for TruthGPT models."""
    
    def __init__(self, config: TruthGPTCompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Quantization state
        self.quantized_model = None
        self.quantization_stats = {}
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize TruthGPT model."""
        self.logger.info(f"🔧 Quantizing TruthGPT model with {self.config.quantization_type} quantization")
        
        if self.config.quantization_type == "dynamic":
            return self._dynamic_quantization(model)
        elif self.config.quantization_type == "static":
            return self._static_quantization(model)
        elif self.config.quantization_type == "qat":
            return self._quantization_aware_training(model)
        else:
            raise ValueError(f"Unknown quantization type: {self.config.quantization_type}")
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        # Select layers to quantize
        layers_to_quantize = {nn.Linear, nn.Conv2d, nn.Conv1d}
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            layers_to_quantize, 
            dtype=torch.qint8
        )
        
        self.logger.info("✅ Dynamic quantization applied")
        return quantized_model
    
    def _static_quantization(self, model: nn.Module) -> nn.Module:
        """Apply static quantization."""
        # Set model to evaluation mode
        model.eval()
        
        # Create quantization configuration
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model.qconfig = qconfig
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(model)
        
        # Calibrate model (would need calibration data in practice)
        # For demo, we'll skip calibration
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        self.logger.info("✅ Static quantization applied")
        return quantized_model
    
    def _quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Apply quantization aware training."""
        # Set quantization configuration
        qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model.qconfig = qconfig
        
        # Prepare model for QAT
        prepared_model = torch.quantization.prepare_qat(model)
        
        self.logger.info("✅ Quantization aware training prepared")
        return prepared_model
    
    def get_quantization_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get quantization statistics."""
        # Calculate model size
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Calculate compression ratio
        compression_ratio = 1.0  # Would be calculated based on actual quantization
        
        return {
            'original_size_mb': original_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'quantization_bits': self.config.quantization_bits
        }

class TruthGPTPruner:
    """Advanced pruner for TruthGPT models."""
    
    def __init__(self, config: TruthGPTCompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Pruning state
        self.pruned_model = None
        self.pruning_stats = {}
    
    def prune_model(self, model: nn.Module) -> nn.Module:
        """Prune TruthGPT model."""
        self.logger.info(f"🔧 Pruning TruthGPT model with {self.config.pruning_type} pruning")
        
        if self.config.pruning_type == "unstructured":
            return self._unstructured_pruning(model)
        elif self.config.pruning_type == "structured":
            return self._structured_pruning(model)
        elif self.config.pruning_type == "global":
            return self._global_pruning(model)
        else:
            raise ValueError(f"Unknown pruning type: {self.config.pruning_type}")
    
    def _unstructured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply unstructured pruning."""
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            self.logger.warning("No parameters found for pruning")
            return model
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_ratio,
        )
        
        # Remove pruning reparametrization
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        
        self.logger.info(f"✅ Unstructured pruning applied with ratio {self.config.pruning_ratio}")
        return model
    
    def _structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning."""
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            self.logger.warning("No parameters found for pruning")
            return model
        
        # Apply structured pruning
        for module, name in parameters_to_prune:
            prune.ln_structured(
                module, 
                name, 
                amount=self.config.pruning_ratio, 
                n=2, 
                dim=0
            )
        
        # Remove pruning reparametrization
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        
        self.logger.info(f"✅ Structured pruning applied with ratio {self.config.pruning_ratio}")
        return model
    
    def _global_pruning(self, model: nn.Module) -> nn.Module:
        """Apply global pruning."""
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            self.logger.warning("No parameters found for pruning")
            return model
        
        # Apply global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_ratio,
        )
        
        # Remove pruning reparametrization
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        
        self.logger.info(f"✅ Global pruning applied with ratio {self.config.pruning_ratio}")
        return model
    
    def get_pruning_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get pruning statistics."""
        # Calculate model size
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        
        return {
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'pruning_ratio': zero_params / total_params if total_params > 0 else 0,
            'compression_ratio': 1.0 - (zero_params / total_params) if total_params > 0 else 1.0
        }

class TruthGPTDistiller:
    """Advanced knowledge distiller for TruthGPT models."""
    
    def __init__(self, config: TruthGPTCompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Distillation state
        self.teacher_model = None
        self.student_model = None
        self.distillation_stats = {}
    
    def load_teacher_model(self, model_path: str) -> nn.Module:
        """Load teacher model for distillation."""
        if not model_path:
            raise ValueError("Teacher model path not provided")
        
        # Load teacher model
        teacher_model = torch.load(model_path, map_location='cpu')
        teacher_model.eval()
        
        self.teacher_model = teacher_model
        self.logger.info(f"✅ Teacher model loaded from {model_path}")
        
        return teacher_model
    
    def distill_model(self, student_model: nn.Module, 
                     data_loader: torch.utils.data.DataLoader,
                     loss_fn: Callable, 
                     optimizer: torch.optim.Optimizer,
                     epochs: int = 10) -> nn.Module:
        """Distill knowledge from teacher to student."""
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded")
        
        self.logger.info(f"🔧 Distilling knowledge for {epochs} epochs")
        
        student_model.train()
        self.teacher_model.eval()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(data)
                
                # Get student predictions
                student_outputs = student_model(data)
                
                # Calculate distillation loss
                distillation_loss = self._calculate_distillation_loss(
                    student_outputs, 
                    teacher_outputs, 
                    target, 
                    loss_fn
                )
                
                # Backward pass
                optimizer.zero_grad()
                distillation_loss.backward()
                optimizer.step()
                
                total_loss += distillation_loss.item()
            
            avg_loss = total_loss / len(data_loader)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        self.logger.info("✅ Knowledge distillation completed")
        return student_model
    
    def _calculate_distillation_loss(self, student_outputs: torch.Tensor,
                                   teacher_outputs: torch.Tensor,
                                   target: torch.Tensor,
                                   loss_fn: Callable) -> torch.Tensor:
        """Calculate distillation loss."""
        # Soft targets loss
        soft_targets_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(student_outputs / self.config.distillation_temperature, dim=1),
            torch.softmax(teacher_outputs / self.config.distillation_temperature, dim=1)
        ) * (self.config.distillation_temperature ** 2)
        
        # Hard targets loss
        hard_targets_loss = loss_fn(student_outputs, target)
        
        # Combine losses
        total_loss = (self.config.distillation_alpha * soft_targets_loss + 
                     (1 - self.config.distillation_alpha) * hard_targets_loss)
        
        return total_loss
    
    def get_distillation_stats(self) -> Dict[str, Any]:
        """Get distillation statistics."""
        return {
            'distillation_temperature': self.config.distillation_temperature,
            'distillation_alpha': self.config.distillation_alpha,
            'teacher_model_loaded': self.teacher_model is not None
        }

class TruthGPTLowRankCompressor:
    """Advanced low-rank compressor for TruthGPT models."""
    
    def __init__(self, config: TruthGPTCompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Compression state
        self.compressed_model = None
        self.compression_stats = {}
    
    def compress_model(self, model: nn.Module) -> nn.Module:
        """Compress TruthGPT model using low-rank decomposition."""
        self.logger.info(f"🔧 Compressing TruthGPT model with low-rank decomposition")
        
        compressed_model = copy.deepcopy(model)
        
        # Apply low-rank compression to linear layers
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                compressed_module = self._compress_linear_layer(module)
                # Replace module
                parent = compressed_model
                for attr in name.split('.')[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, name.split('.')[-1], compressed_module)
        
        self.logger.info("✅ Low-rank compression applied")
        return compressed_model
    
    def _compress_linear_layer(self, layer: nn.Linear) -> nn.Module:
        """Compress a linear layer using low-rank decomposition."""
        # Get weight matrix
        weight = layer.weight.data
        
        # Perform SVD
        U, S, V = torch.svd(weight)
        
        # Calculate rank
        original_rank = min(weight.shape)
        target_rank = max(1, int(original_rank * self.config.rank_ratio))
        
        # Truncate SVD
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        V_truncated = V[:target_rank, :]
        
        # Create compressed layers
        compressed_layer = nn.Sequential(
            nn.Linear(layer.in_features, target_rank, bias=False),
            nn.Linear(target_rank, layer.out_features, bias=layer.bias is not None)
        )
        
        # Set weights
        compressed_layer[0].weight.data = U_truncated.t()
        compressed_layer[1].weight.data = torch.diag(S_truncated) @ V_truncated
        
        if layer.bias is not None:
            compressed_layer[1].bias.data = layer.bias.data
        
        return compressed_layer
    
    def get_compression_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get compression statistics."""
        # Calculate model size
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate compression ratio
        compression_ratio = self.config.rank_ratio
        
        return {
            'total_parameters': total_params,
            'compression_ratio': compression_ratio,
            'rank_ratio': self.config.rank_ratio
        }

class TruthGPTCompressionManager:
    """Advanced compression manager for TruthGPT models."""
    
    def __init__(self, config: TruthGPTCompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Compression components
        self.quantizer = TruthGPTQuantizer(config) if config.enable_quantization else None
        self.pruner = TruthGPTPruner(config) if config.enable_pruning else None
        self.distiller = TruthGPTDistiller(config) if config.enable_distillation else None
        self.low_rank_compressor = TruthGPTLowRankCompressor(config) if config.enable_low_rank else None
        
        # Compression state
        self.compressed_model = None
        self.compression_stats = {}
    
    def compress_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive compression to TruthGPT model."""
        self.logger.info("🚀 Starting comprehensive TruthGPT model compression")
        
        compressed_model = model
        
        # Apply quantization
        if self.quantizer:
            compressed_model = self.quantizer.quantize_model(compressed_model)
        
        # Apply pruning
        if self.pruner:
            compressed_model = self.pruner.prune_model(compressed_model)
        
        # Apply low-rank compression
        if self.low_rank_compressor:
            compressed_model = self.low_rank_compressor.compress_model(compressed_model)
        
        # Apply distillation
        if self.distiller and self.distiller.teacher_model:
            # This would require training data and optimizer
            self.logger.info("Distillation requires training setup")
        
        self.compressed_model = compressed_model
        self.logger.info("✅ TruthGPT model compression completed")
        
        return compressed_model
    
    def get_compression_stats(self, original_model: nn.Module, 
                            compressed_model: nn.Module) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        # Calculate original model size
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
        
        # Calculate compression ratio
        compression_ratio = compressed_size / original_size
        
        # Get individual compression stats
        stats = {
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction': 1.0 - compression_ratio
        }
        
        # Add individual compression stats
        if self.quantizer:
            stats.update(self.quantizer.get_quantization_stats(compressed_model))
        
        if self.pruner:
            stats.update(self.pruner.get_pruning_stats(compressed_model))
        
        if self.low_rank_compressor:
            stats.update(self.low_rank_compressor.get_compression_stats(compressed_model))
        
        return stats

# Factory functions
def create_truthgpt_compression_manager(config: TruthGPTCompressionConfig) -> TruthGPTCompressionManager:
    """Create TruthGPT compression manager."""
    return TruthGPTCompressionManager(config)

def compress_truthgpt_model(model: nn.Module, config: TruthGPTCompressionConfig) -> nn.Module:
    """Quick compress TruthGPT model."""
    manager = create_truthgpt_compression_manager(config)
    return manager.compress_model(model)

# Example usage
if __name__ == "__main__":
    # Example TruthGPT model compression
    print("🚀 TruthGPT Model Compression Demo")
    print("=" * 50)
    
    # Create a sample TruthGPT-style model
    class TruthGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 768)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(768, 12, 3072, dropout=0.1),
                num_layers=12
            )
            self.lm_head = nn.Linear(768, 10000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.lm_head(x)
            return x
    
    # Create model
    model = TruthGPTModel()
    
    # Create compression configuration
    config = TruthGPTCompressionConfig(
        enable_quantization=True,
        enable_pruning=True,
        enable_low_rank=True,
        quantization_type="dynamic",
        pruning_ratio=0.1,
        rank_ratio=0.5
    )
    
    # Create compression manager
    manager = create_truthgpt_compression_manager(config)
    
    # Compress model
    compressed_model = manager.compress_model(model)
    
    # Get compression stats
    stats = manager.get_compression_stats(model, compressed_model)
    print(f"Compression stats: {stats}")
    
    print("✅ TruthGPT model compression completed!")



