"""
Boosting Ensemble
=================

Boosting-based ensemble methods.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any
from .config import EnsembleConfig, EnsembleStrategy
from .base import BaseModel

logger = logging.getLogger(__name__)

class BoostingEnsemble:
    """Boosting ensemble implementation"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = []
        self.weights = []
        self.training_history = []
        logger.info("✅ Boosting Ensemble initialized")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train boosting ensemble"""
        logger.info("🏋️ Training boosting ensemble")
        
        # Initialize weights
        sample_weights = np.ones(len(X)) / len(X)
        
        model_results = []
        
        for i in range(self.config.boosting_iterations):
            # Create model
            model = BaseModel(i, "neural_network", self.config)
            model.initialize_model()
            
            # Train model with weighted samples
            result = self._train_weighted_model(model, X, y, sample_weights)
            model_results.append(result)
            
            # Store model
            self.models.append(model)
            
            # Calculate model weight
            model_weight = self._calculate_model_weight(model, X, y, sample_weights)
            self.weights.append(model_weight)
            
            # Update sample weights
            sample_weights = self._update_sample_weights(model, X, y, sample_weights, model_weight)
        
        training_result = {
            'strategy': EnsembleStrategy.BOOSTING_ENSEMBLE.value,
            'boosting_method': self.config.boosting_method.value,
            'iterations': self.config.boosting_iterations,
            'num_models': len(self.models),
            'model_weights': self.weights,
            'model_results': model_results,
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def _train_weighted_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray, 
                             sample_weights: np.ndarray) -> Dict[str, Any]:
        """Train model with weighted samples"""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        weights_tensor = torch.FloatTensor(sample_weights)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(model.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Training loop
        num_epochs = 5
        losses = []
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            weighted_loss = (loss * weights_tensor).mean()
            weighted_loss.backward()
            optimizer.step()
            losses.append(weighted_loss.item())
        
        return {
            'epochs': num_epochs,
            'final_loss': losses[-1],
            'losses': losses
        }
    
    def _calculate_model_weight(self, model: BaseModel, X: np.ndarray, y: np.ndarray, 
                               sample_weights: np.ndarray) -> float:
        """Calculate model weight"""
        # Get predictions
        predictions = model.predict(X)
        
        if len(predictions.shape) == 1:
            pred_classes = (predictions > 0.5).astype(int)
        else:
            pred_classes = np.argmax(predictions, axis=1)
        
        # Calculate weighted error
        errors = (pred_classes != y).astype(float)
        weighted_error = np.sum(errors * sample_weights) / np.sum(sample_weights)
        
        # Calculate model weight
        if weighted_error == 0:
            model_weight = 1.0
        else:
            model_weight = 0.5 * np.log((1 - weighted_error) / weighted_error)
        
        return model_weight
    
    def _update_sample_weights(self, model: BaseModel, X: np.ndarray, y: np.ndarray, 
                              sample_weights: np.ndarray, model_weight: float) -> np.ndarray:
        """Update sample weights"""
        # Get predictions
        predictions = model.predict(X)
        
        if len(predictions.shape) == 1:
            pred_classes = (predictions > 0.5).astype(int)
        else:
            pred_classes = np.argmax(predictions, axis=1)
        
        # Update weights
        errors = (pred_classes != y).astype(float)
        new_weights = sample_weights * np.exp(model_weight * errors)
        
        # Normalize weights
        new_weights = new_weights / np.sum(new_weights)
        
        return new_weights
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += self.weights[i] * pred
        
        if len(weighted_pred.shape) == 1:
            return (weighted_pred > 0.5).astype(int)
        else:
            return np.argmax(weighted_pred, axis=1)

