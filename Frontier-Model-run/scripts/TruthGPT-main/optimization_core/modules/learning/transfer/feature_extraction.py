"""
Feature Extraction
==================

Feature extraction logic.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from .config import TransferLearningConfig
from .enums import TransferStrategy

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Feature extraction implementation"""
    
    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.feature_extractor = None
        self.training_history = []
        logger.info("✅ Feature Extractor initialized")
    
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
    
    def extract_features(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Extract features from model"""
        logger.info("🔍 Extracting features")
        
        # Create feature extractor if not exists
        if self.feature_extractor is None:
            self.feature_extractor = self.create_feature_extractor()
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(data)
        
        return features
    
    def train_feature_extractor(self, model: nn.Module, train_data: torch.Tensor, 
                               train_labels: torch.Tensor, num_epochs: int = 10) -> Dict[str, Any]:
        """Train feature extractor"""
        logger.info("🏋️ Training feature extractor")
        
        # Create feature extractor
        self.feature_extractor = self.create_feature_extractor()
        
        # Create classifier
        classifier = nn.Sequential(
            nn.Linear(self.config.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.config.num_classes)
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(classifier.parameters()),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        training_accuracies = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Forward pass
            features = self.feature_extractor(train_data)
            outputs = classifier(features)
            loss = criterion(outputs, train_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == train_labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            
            training_losses.append(epoch_loss)
            training_accuracies.append(epoch_accuracy)
            
            if epoch % 5 == 0:
                logger.info(f"   Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")
        
        training_result = {
            'strategy': TransferStrategy.FEATURE_EXTRACTION.value,
            'epochs': num_epochs,
            'training_losses': training_losses,
            'training_accuracies': training_accuracies,
            'final_loss': training_losses[-1],
            'final_accuracy': training_accuracies[-1],
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result

