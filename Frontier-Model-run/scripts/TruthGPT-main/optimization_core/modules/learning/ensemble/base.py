"""
Base Model
==========

Base model wrapper for ensemble members.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from .config import EnsembleConfig

logger = logging.getLogger(__name__)

class BaseModel:
    """Base model for ensemble learning"""
    
    def __init__(self, model_id: int, model_type: str, config: EnsembleConfig):
        self.model_id = model_id
        self.model_type = model_type
        self.config = config
        self.model = None
        self.training_history = []
        logger.info(f"✅ Base Model {model_id} ({model_type}) initialized")
    
    def create_neural_network(self) -> nn.Module:
        """Create neural network model"""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        
        return model
    
    def create_random_forest(self):
        """Create random forest model"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def create_svm(self):
        """Create SVM model"""
        return SVC(probability=True, random_state=42)
    
    def initialize_model(self):
        """Initialize model based on type"""
        if self.model_type == "neural_network":
            self.model = self.create_neural_network()
        elif self.model_type == "random_forest":
            self.model = self.create_random_forest()
        elif self.model_type == "svm":
            self.model = self.create_svm()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train model"""
        logger.info(f"🏋️ Training model {self.model_id} ({self.model_type})")
        
        # Initialize model if not exists
        if self.model is None:
            self.initialize_model()
        
        training_result = {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'training_samples': len(X),
            'status': 'success'
        }
        
        if self.model_type == "neural_network":
            training_result.update(self._train_neural_network(X, y))
        else:
            training_result.update(self._train_sklearn_model(X, y))
        
        self.training_history.append(training_result)
        return training_result
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train neural network"""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        num_epochs = 10
        losses = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return {
            'epochs': num_epochs,
            'final_loss': losses[-1],
            'losses': losses
        }
    
    def _train_sklearn_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train sklearn model"""
        # Train model
        self.model.fit(X, y)
        
        # Calculate accuracy
        accuracy = self.model.score(X, y)
        
        return {
            'accuracy': accuracy,
            'model_params': str(self.model.get_params())
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        if self.model_type == "neural_network":
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = self.model(X_tensor)
                predictions = torch.softmax(outputs, dim=1).numpy()
        else:
            if hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X)
            else:
                predictions = self.model.predict(X)
        
        return predictions
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence"""
        predictions = self.predict(X)
        
        if len(predictions.shape) == 1:
            # Binary classification
            confidence = np.abs(predictions - 0.5) * 2
        else:
            # Multi-class classification
            confidence = np.max(predictions, axis=1)
        
        return confidence

