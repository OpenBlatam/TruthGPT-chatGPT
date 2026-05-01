"""
Dynamic Ensemble
================

Dynamic ensemble selection methods.
"""
import numpy as np
import logging
from typing import Dict, Any
from .config import EnsembleConfig, EnsembleStrategy
from .base import BaseModel

logger = logging.getLogger(__name__)

class DynamicEnsemble:
    """Dynamic ensemble implementation"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = []
        self.performance_history = []
        self.training_history = []
        logger.info("✅ Dynamic Ensemble initialized")
    
    def add_model(self, model: BaseModel):
        """Add model to ensemble"""
        self.models.append(model)
        self.performance_history.append([])
        logger.info(f"➕ Added model {model.model_id} to dynamic ensemble")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train dynamic ensemble"""
        logger.info("🏋️ Training dynamic ensemble")
        
        # Train individual models
        model_results = []
        for model in self.models:
            result = model.train(X, y)
            model_results.append(result)
        
        # Evaluate model performance
        self._evaluate_models(X, y)
        
        training_result = {
            'strategy': EnsembleStrategy.DYNAMIC_ENSEMBLE.value,
            'num_models': len(self.models),
            'model_results': model_results,
            'performance_history': self.performance_history,
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def _evaluate_models(self, X: np.ndarray, y: np.ndarray):
        """Evaluate model performance"""
        for i, model in enumerate(self.models):
            # Get predictions
            predictions = model.predict(X)
            
            if len(predictions.shape) == 1:
                pred_classes = (predictions > 0.5).astype(int)
            else:
                pred_classes = np.argmax(predictions, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(pred_classes == y)
            self.performance_history[i].append(accuracy)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make dynamic ensemble predictions"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        weights = []
        
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred)
            
            # Calculate weight based on recent performance
            if self.performance_history[i]:
                recent_performance = np.mean(self.performance_history[i][-5:])  # Last 5 evaluations
                weights.append(recent_performance)
            else:
                weights.append(1.0)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted prediction
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += weights[i] * pred
        
        if len(weighted_pred.shape) == 1:
            return (weighted_pred > 0.5).astype(int)
        else:
            return np.argmax(weighted_pred, axis=1)

