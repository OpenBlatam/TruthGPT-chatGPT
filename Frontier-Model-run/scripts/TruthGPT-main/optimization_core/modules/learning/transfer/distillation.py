"""
Knowledge Distillation
======================

Knowledge distillation implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any
from .config import TransferLearningConfig
from .enums import TransferStrategy

logger = logging.getLogger(__name__)

class KnowledgeDistiller:
    """Knowledge distillation implementation"""
    
    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.teacher_model = None
        self.student_model = None
        self.training_history = []
        logger.info("✅ Knowledge Distiller initialized")
    
    def create_teacher_model(self) -> nn.Module:
        """Create teacher model"""
        teacher = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.config.feature_dim),
            nn.ReLU(),
            nn.Linear(self.config.feature_dim, self.config.num_classes)
        )
        
        return teacher
    
    def create_student_model(self) -> nn.Module:
        """Create student model"""
        student = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.config.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.feature_dim // 2, self.config.num_classes)
        )
        
        return student
    
    def distill_knowledge(self, teacher_data: torch.Tensor, teacher_labels: torch.Tensor,
                         student_data: torch.Tensor, student_labels: torch.Tensor,
                         num_epochs: int = 10) -> Dict[str, Any]:
        """Distill knowledge from teacher to student"""
        logger.info("🎓 Distilling knowledge from teacher to student")
        
        # Create models
        self.teacher_model = self.create_teacher_model()
        self.student_model = self.create_student_model()
        
        # Train teacher model
        teacher_optimizer = torch.optim.Adam(self.teacher_model.parameters(), lr=self.config.learning_rate)
        teacher_criterion = nn.CrossEntropyLoss()
        
        logger.info("👨‍🏫 Training teacher model")
        for epoch in range(num_epochs):
            teacher_optimizer.zero_grad()
            teacher_outputs = self.teacher_model(teacher_data)
            teacher_loss = teacher_criterion(teacher_outputs, teacher_labels)
            teacher_loss.backward()
            teacher_optimizer.step()
        
        # Distill knowledge to student
        student_optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.config.learning_rate)
        
        distillation_losses = []
        student_accuracies = []
        
        logger.info("👨‍🎓 Training student model with distillation")
        for epoch in range(num_epochs):
            # Teacher predictions
            with torch.no_grad():
                teacher_outputs = self.teacher_model(student_data)
                teacher_soft = F.softmax(teacher_outputs / self.config.temperature, dim=1)
            
            # Student predictions
            student_outputs = self.student_model(student_data)
            student_soft = F.log_softmax(student_outputs / self.config.temperature, dim=1)
            
            # Distillation loss
            distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
            
            # Student loss
            student_loss = F.cross_entropy(student_outputs, student_labels)
            
            # Combined loss
            total_loss = self.config.alpha * distillation_loss + self.config.beta * student_loss
            
            # Backward pass
            student_optimizer.zero_grad()
            total_loss.backward()
            student_optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(student_outputs.data, 1)
            accuracy = (predicted == student_labels).float().mean()
            
            distillation_losses.append(distillation_loss.item())
            student_accuracies.append(accuracy.item())
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Distillation Loss = {distillation_loss.item():.4f}, Accuracy = {accuracy.item():.4f}")
        
        distillation_result = {
            'strategy': TransferStrategy.KNOWLEDGE_DISTILLATION.value,
            'distillation_type': self.config.distillation_type.value,
            'epochs': num_epochs,
            'distillation_losses': distillation_losses,
            'student_accuracies': student_accuracies,
            'final_distillation_loss': distillation_losses[-1],
            'final_accuracy': student_accuracies[-1],
            'temperature': self.config.temperature,
            'alpha': self.config.alpha,
            'beta': self.config.beta,
            'status': 'success'
        }
        
        self.training_history.append(distillation_result)
        return distillation_result

