"""
Domain Adaptation
=================

Domain adaptation logic.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from .config import TransferLearningConfig
from .enums import TransferStrategy, DomainAdaptationMethod

logger = logging.getLogger(__name__)

class DomainAdapter:
    """Domain adaptation implementation"""
    
    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.source_model = None
        self.target_model = None
        self.domain_classifier = None
        self.training_history = []
        logger.info("✅ Domain Adapter initialized")
    
    def create_domain_classifier(self) -> nn.Module:
        """Create domain classifier"""
        domain_classifier = nn.Sequential(
            nn.Linear(self.config.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 domains: source and target
        )
        return domain_classifier
    
    def create_feature_extractor(self) -> nn.Module:
        """Create feature extractor"""
        feature_extractor = nn.Sequential(
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
            nn.Linear(256, self.config.feature_dim)
        )
        return feature_extractor
    
    def create_task_classifier(self) -> nn.Module:
        """Create task classifier"""
        task_classifier = nn.Sequential(
            nn.Linear(self.config.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.config.num_classes)
        )
        return task_classifier
    
    def adapt_domain(self, source_data: torch.Tensor, source_labels: torch.Tensor,
                    target_data: torch.Tensor, num_epochs: int = 10) -> Dict[str, Any]:
        """Adapt domain from source to target"""
        logger.info("🔄 Adapting domain from source to target")
        
        # Create models
        feature_extractor = self.create_feature_extractor()
        task_classifier = self.create_task_classifier()
        domain_classifier = self.create_domain_classifier()
        
        # Create optimizers
        feature_optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=self.config.learning_rate)
        task_optimizer = torch.optim.Adam(task_classifier.parameters(), lr=self.config.learning_rate)
        domain_optimizer = torch.optim.Adam(domain_classifier.parameters(), lr=self.config.learning_rate)
        
        # Loss functions
        task_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        # Create domain labels
        source_domain_labels = torch.zeros(source_data.shape[0], dtype=torch.long)
        target_domain_labels = torch.ones(target_data.shape[0], dtype=torch.long)
        
        adaptation_losses = []
        task_accuracies = []
        domain_accuracies = []
        
        for epoch in range(num_epochs):
            # Extract features
            source_features = feature_extractor(source_data)
            target_features = feature_extractor(target_data)
            
            # Task classification (source only)
            source_task_outputs = task_classifier(source_features)
            task_loss = task_criterion(source_task_outputs, source_labels)
            
            # Domain classification
            all_features = torch.cat([source_features, target_features], dim=0)
            all_domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
            
            domain_outputs = domain_classifier(all_features)
            domain_loss = domain_criterion(domain_outputs, all_domain_labels)
            
            # Adversarial training
            if self.config.domain_adaptation_method == DomainAdaptationMethod.DANN:
                # Gradient reversal layer (simplified)
                adversarial_loss = -self.config.adversarial_weight * domain_loss
                total_loss = task_loss + adversarial_loss
            else:
                total_loss = task_loss + self.config.domain_loss_weight * domain_loss
            
            # Backward pass
            feature_optimizer.zero_grad()
            task_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            
            # Retain graph for second backward pass if needed, or structured differently
            total_loss.backward()
            
            feature_optimizer.step()
            task_optimizer.step()
            domain_optimizer.step()
            
            # Calculate accuracies
            _, task_predicted = torch.max(source_task_outputs.data, 1)
            task_accuracy = (task_predicted == source_labels).float().mean()
            
            _, domain_predicted = torch.max(domain_outputs.data, 1)
            domain_accuracy = (domain_predicted == all_domain_labels).float().mean()
            
            adaptation_losses.append(total_loss.item())
            task_accuracies.append(task_accuracy.item())
            domain_accuracies.append(domain_accuracy.item())
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Task Loss = {task_loss.item():.4f}, Domain Loss = {domain_loss.item():.4f}")
        
        adaptation_result = {
            'strategy': TransferStrategy.DOMAIN_ADAPTATION.value,
            'domain_adaptation_method': self.config.domain_adaptation_method.value,
            'epochs': num_epochs,
            'adaptation_losses': adaptation_losses,
            'task_accuracies': task_accuracies,
            'domain_accuracies': domain_accuracies,
            'final_task_accuracy': task_accuracies[-1],
            'final_domain_accuracy': domain_accuracies[-1],
            'domain_loss_weight': self.config.domain_loss_weight,
            'adversarial_weight': self.config.adversarial_weight,
            'status': 'success'
        }
        
        self.training_history.append(adaptation_result)
        return adaptation_result

