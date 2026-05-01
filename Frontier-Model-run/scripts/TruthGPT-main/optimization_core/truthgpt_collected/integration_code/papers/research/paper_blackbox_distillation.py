#!/usr/bin/env python3
"""
Black-Box On-Policy Distillation of Large Language Models
==========================================================

Proponen un método para destilar ("entrenar un modelo estudiante") sin acceso
a los logits del modelo "maestro", solo mediante sus salidas, usando un enfoque adversarial.

Técnica principal: Adversarial distillation from black-box teacher model.

Basado en: Hugging Face / arXiv paper (November 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlackBoxDistillationConfig:
    """Configuración para Black-Box Distillation."""
    hidden_dim: int = 512
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for distillation loss
    use_adversarial: bool = True
    discriminator_hidden_dim: int = 256
    adversarial_weight: float = 0.1
    use_attention_distillation: bool = True


class AdversarialDiscriminator(nn.Module):
    """
    Discriminador adversarial para distinguir entre outputs del teacher y student.
    
    Técnica: Usa un discriminador para mejorar la calidad de la distillation.
    """
    
    def __init__(self, config: BlackBoxDistillationConfig):
        super().__init__()
        self.config = config
        
        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.discriminator_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(config.discriminator_hidden_dim, config.discriminator_hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(config.discriminator_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        for module in self.discriminator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        logger.info("Initialized AdversarialDiscriminator")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between teacher and student outputs.
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            
        Returns:
            scores: [batch, seq] probability of being from teacher
        """
        # Use last token for discrimination
        last_token = hidden_states[:, -1, :]  # [batch, hidden_dim]
        scores = self.discriminator(last_token).squeeze(-1)  # [batch]
        return scores


class BlackBoxDistillation(nn.Module):
    """
    Black-Box Distillation module.
    
    Técnica: Distilla conocimiento de un modelo teacher (black-box)
    usando solo sus outputs, sin acceso a logits.
    """
    
    def __init__(self, config: BlackBoxDistillationConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.temperature = config.temperature
        self.alpha = config.alpha
        
        # Adversarial discriminator
        if config.use_adversarial:
            self.discriminator = AdversarialDiscriminator(config)
        else:
            self.discriminator = None
        
        # Attention distillation (if enabled)
        if config.use_attention_distillation:
            self.attention_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            nn.init.xavier_uniform_(self.attention_proj.weight)
        else:
            self.attention_proj = None
        
        # Metrics
        self.register_buffer('distillation_loss', torch.tensor(0.0))
        self.register_buffer('adversarial_loss', torch.tensor(0.0))
        self.register_buffer('discriminator_accuracy', torch.tensor(0.5))
        self.register_buffer('student_teacher_similarity', torch.tensor(0.0))
        
        logger.info(f"Initialized BlackBoxDistillation: temperature={config.temperature}, "
                   f"adversarial={config.use_adversarial}")
    
    def compute_distillation_loss(self, student_output: torch.Tensor, 
                                  teacher_output: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss from teacher outputs (no logits needed).
        
        Args:
            student_output: [batch, seq, hidden_dim]
            teacher_output: [batch, seq, hidden_dim]
            
        Returns:
            loss: Distillation loss
        """
        # Use cosine similarity as proxy for distribution matching
        student_norm = F.normalize(student_output, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_output, p=2, dim=-1)
        
        # Cosine similarity
        similarity = (student_norm * teacher_norm).sum(dim=-1)  # [batch, seq]
        
        # Convert to loss (maximize similarity = minimize (1 - similarity))
        distillation_loss = (1.0 - similarity).mean()
        
        # Temperature scaling
        distillation_loss = distillation_loss / self.temperature
        
        # Update metrics
        self.student_teacher_similarity = 0.9 * self.student_teacher_similarity + 0.1 * similarity.mean().item()
        self.distillation_loss = 0.9 * self.distillation_loss + 0.1 * distillation_loss.item()
        
        return distillation_loss
    
    def compute_adversarial_loss(self, student_output: torch.Tensor,
                                teacher_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial loss for distillation.
        
        Args:
            student_output: [batch, seq, hidden_dim]
            teacher_output: [batch, seq, hidden_dim]
            
        Returns:
            student_loss: Loss for student (trying to fool discriminator)
            discriminator_loss: Loss for discriminator
        """
        if self.discriminator is None:
            return torch.tensor(0.0, device=student_output.device), torch.tensor(0.0, device=student_output.device)
        
        # Discriminate teacher outputs (should be 1)
        teacher_scores = self.discriminator(teacher_output)  # [batch]
        teacher_labels = torch.ones_like(teacher_scores)
        teacher_d_loss = F.binary_cross_entropy(teacher_scores, teacher_labels)
        
        # Discriminate student outputs (should be 0, but student tries to make it 1)
        student_scores = self.discriminator(student_output)  # [batch]
        student_labels = torch.zeros_like(student_scores)
        student_d_loss = F.binary_cross_entropy(student_scores, student_labels)
        
        # Discriminator loss (wants to distinguish correctly)
        discriminator_loss = teacher_d_loss + student_d_loss
        
        # Student loss (wants to fool discriminator - make student look like teacher)
        student_adv_loss = F.binary_cross_entropy(student_scores, torch.ones_like(student_scores))
        
        # Update metrics
        accuracy = ((teacher_scores > 0.5).float().mean() + (student_scores < 0.5).float().mean()) / 2.0
        self.discriminator_accuracy = 0.9 * self.discriminator_accuracy + 0.1 * accuracy.item()
        self.adversarial_loss = 0.9 * self.adversarial_loss + 0.1 * student_adv_loss.item()
        
        return student_adv_loss, discriminator_loss
    
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass: compute distillation loss.
        
        Args:
            student_output: [batch, seq, hidden_dim]
            teacher_output: [batch, seq, hidden_dim]
            
        Returns:
            total_loss: Total distillation loss
            metadata: Dict with loss components
        """
        # Distillation loss
        distillation_loss = self.compute_distillation_loss(student_output, teacher_output)
        
        # Adversarial loss
        if self.config.use_adversarial:
            student_adv_loss, discriminator_loss = self.compute_adversarial_loss(student_output, teacher_output)
            total_loss = self.alpha * distillation_loss + self.config.adversarial_weight * student_adv_loss
        else:
            student_adv_loss = torch.tensor(0.0, device=student_output.device)
            discriminator_loss = torch.tensor(0.0, device=student_output.device)
            total_loss = distillation_loss
        
        metadata = {
            'distillation_loss': distillation_loss.item(),
            'adversarial_loss': student_adv_loss.item() if isinstance(student_adv_loss, torch.Tensor) else student_adv_loss,
            'discriminator_loss': discriminator_loss.item() if isinstance(discriminator_loss, torch.Tensor) else discriminator_loss,
            'total_loss': total_loss.item(),
            'similarity': self.student_teacher_similarity.item()
        }
        
        return total_loss, metadata
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get distillation metrics."""
        return {
            'distillation_loss': self.distillation_loss.item(),
            'adversarial_loss': self.adversarial_loss.item(),
            'discriminator_accuracy': self.discriminator_accuracy.item(),
            'student_teacher_similarity': self.student_teacher_similarity.item(),
            'temperature': self.temperature,
            'use_adversarial': self.config.use_adversarial
        }


class BlackBoxDistillationModule(nn.Module):
    """
    Módulo completo de Black-Box Distillation.
    """
    
    def __init__(self, config: BlackBoxDistillationConfig):
        super().__init__()
        self.config = config
        self.distillation = BlackBoxDistillation(config)
        
        logger.info(f"Initialized BlackBoxDistillationModule with config: {config}")
    
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass."""
        return self.distillation(student_output, teacher_output)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics."""
        return self.distillation.get_metrics()


class TruthGPT_BlackBoxDistillation_Integration(nn.Module):
    """Integración de Black-Box Distillation con TruthGPT."""
    
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module, 
                 distillation_config: BlackBoxDistillationConfig):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.distillation_module = BlackBoxDistillationModule(distillation_config)
    
    def forward(self, *args, **kwargs):
        """Forward pass con distillation."""
        # Get outputs from both models
        student_output = self.student_model(*args, **kwargs)
        with torch.no_grad():
            teacher_output = self.teacher_model(*args, **kwargs)
        
        # Compute distillation loss
        distillation_loss, metadata = self.distillation_module(student_output, teacher_output)
        
        return student_output, distillation_loss, metadata


if __name__ == "__main__":
    config = BlackBoxDistillationConfig(
        hidden_dim=512,
        temperature=4.0,
        use_adversarial=True
    )
    
    # Create dummy models
    class DummyModel(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        def forward(self, x):
            return self.proj(x)
    
    student = DummyModel(config.hidden_dim)
    teacher = DummyModel(config.hidden_dim)
    
    distillation = BlackBoxDistillationModule(config)
    x = torch.randn(2, 32, config.hidden_dim)
    student_out = student(x)
    teacher_out = teacher(x)
    
    loss, metadata = distillation(student_out, teacher_out)
    metrics = distillation.get_metrics()
    
    print(f"✅ Black-Box Distillation test:")
    print(f"   Distillation loss: {loss.item():.6f}")
    print(f"   Similarity: {metadata['similarity']:.4f}")
    print(f"   Discriminator accuracy: {metrics['discriminator_accuracy']:.4f}")




