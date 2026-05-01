"""
Voting Ensemble
===============

Voting-based ensemble methods.
"""
import numpy as np
import logging
from typing import Dict, Any, List
from .config import EnsembleConfig, EnsembleStrategy, VotingStrategy
from .base import BaseModel

logger = logging.getLogger(__name__)

class VotingEnsemble:
    """Voting ensemble implementation"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = []
        self.weights = []
        self.training_history = []
        logger.info("✅ Voting Ensemble initialized")
    
    def add_model(self, model: BaseModel):
        """Add model to ensemble"""
        self.models.append(model)
        self.weights.append(1.0)  # Initialize with equal weights
        logger.info(f"➕ Added model {model.model_id} to voting ensemble")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train voting ensemble"""
        logger.info("🏋️ Training voting ensemble")
        
        # Train individual models
        model_results = []
        for model in self.models:
            result = model.train(X, y)
            model_results.append(result)
        
        # Learn weights if weighted voting is enabled
        if self.config.enable_weighted_voting:
            self._learn_weights(X, y)
        
        training_result = {
            'strategy': EnsembleStrategy.VOTING_ENSEMBLE.value,
            'voting_strategy': self.config.voting_strategy.value,
            'num_models': len(self.models),
            'model_results': model_results,
            'weights': self.weights,
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def _learn_weights(self, X: np.ndarray, y: np.ndarray):
        """Learn optimal weights for models"""
        logger.info("⚖️ Learning optimal weights")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Simple weight learning based on individual accuracy
        accuracies = []
        for i, model in enumerate(self.models):
            if model.model_type == "neural_network":
                # For neural networks, use validation accuracy (simulated here)
                accuracy = 0.8  # Placeholder
            else:
                accuracy = model.model.score(X, y)
            accuracies.append(accuracy)
        
        # Normalize weights
        total_accuracy = sum(accuracies)
        self.weights = [acc / total_accuracy for acc in accuracies]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Apply voting strategy
        if self.config.voting_strategy == VotingStrategy.HARD_VOTING:
            return self._hard_voting(predictions)
        elif self.config.voting_strategy == VotingStrategy.SOFT_VOTING:
            return self._soft_voting(predictions)
        elif self.config.voting_strategy == VotingStrategy.WEIGHTED_VOTING:
            return self._weighted_voting(predictions)
        else:
            return self._confidence_voting(predictions, X)
    
    def _hard_voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        # Convert probabilities to class predictions
        class_predictions = []
        for pred in predictions:
            if len(pred.shape) == 1:
                class_pred = (pred > 0.5).astype(int)
            else:
                class_pred = np.argmax(pred, axis=1)
            class_predictions.append(class_pred)
        
        # Majority vote
        ensemble_pred = np.array(class_predictions).T
        final_pred = []
        for row in ensemble_pred:
            unique, counts = np.unique(row, return_counts=True)
            final_pred.append(unique[np.argmax(counts)])
        
        return np.array(final_pred)
    
    def _soft_voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        # Average probabilities
        avg_pred = np.mean(predictions, axis=0)
        
        if len(avg_pred.shape) == 1:
            return (avg_pred > 0.5).astype(int)
        else:
            return np.argmax(avg_pred, axis=1)
    
    def _weighted_voting(self, predictions: List[np.ndarray]) -> np.ndarray:
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += self.weights[i] * pred
        
        if len(weighted_pred.shape) == 1:
            return (weighted_pred > 0.5).astype(int)
        else:
            return np.argmax(weighted_pred, axis=1)
    
    def _confidence_voting(self, predictions: List[np.ndarray], X: np.ndarray) -> np.ndarray:
        # Calculate confidence for each model
        confidences = []
        for model in self.models:
            conf = model.get_confidence(X)
            confidences.append(conf)
        
        # Weight by confidence (simplified for implementation)
        # This is a complex area, simplified here
        return self._soft_voting(predictions)

