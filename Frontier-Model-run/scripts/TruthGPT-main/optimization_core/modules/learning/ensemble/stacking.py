"""
Stacking Ensemble
=================

Stacking-based ensemble methods.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from .config import EnsembleConfig, EnsembleStrategy
from .base import BaseModel

logger = logging.getLogger(__name__)

class StackingEnsemble:
    """Stacking ensemble implementation"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.base_models = []
        self.meta_learner = None
        self.training_history = []
        logger.info("✅ Stacking Ensemble initialized")
    
    def add_model(self, model: BaseModel):
        """Add base model to ensemble"""
        self.base_models.append(model)
        logger.info(f"➕ Added model {model.model_id} to stacking ensemble")
    
    def create_meta_learner(self):
        """Create meta-learner"""
        if self.config.meta_learner_type == "logistic_regression":
            return LogisticRegression(random_state=42)
        elif self.config.meta_learner_type == "neural_network":
            return nn.Sequential(
                nn.Linear(len(self.base_models), 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 10)
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {self.config.meta_learner_type}")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train stacking ensemble"""
        logger.info("🏋️ Training stacking ensemble")
        
        # Train base models
        base_results = []
        for model in self.base_models:
            result = model.train(X, y)
            base_results.append(result)
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Train meta-learner
        meta_result = self._train_meta_learner(meta_features, y)
        
        training_result = {
            'strategy': EnsembleStrategy.STACKING_ENSEMBLE.value,
            'num_base_models': len(self.base_models),
            'meta_learner_type': self.config.meta_learner_type,
            'base_results': base_results,
            'meta_result': meta_result,
            'status': 'success'
        }
        
        self.training_history.append(training_result)
        return training_result
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base models"""
        meta_features = []
        
        for model in self.base_models:
            pred = model.predict(X)
            if len(pred.shape) == 1:
                meta_features.append(pred.reshape(-1, 1))
            else:
                meta_features.append(pred)
        
        return np.hstack(meta_features)
    
    def _train_meta_learner(self, meta_features: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train meta-learner"""
        self.meta_learner = self.create_meta_learner()
        
        if isinstance(self.meta_learner, nn.Module):
            # Neural network meta-learner
            X_tensor = torch.FloatTensor(meta_features)
            y_tensor = torch.LongTensor(y)
            
            optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            losses = []
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = self.meta_learner(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            return {
                'epochs': 10,
                'final_loss': losses[-1],
                'losses': losses
            }
        else:
            # Sklearn meta-learner
            self.meta_learner.fit(meta_features, y)
            accuracy = self.meta_learner.score(meta_features, y)
            
            return {
                'accuracy': accuracy,
                'model_params': str(self.meta_learner.get_params())
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.base_models or self.meta_learner is None:
            raise ValueError("Ensemble not trained")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Meta-learner prediction
        if isinstance(self.meta_learner, nn.Module):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(meta_features)
                outputs = self.meta_learner(X_tensor)
                predictions = torch.softmax(outputs, dim=1).numpy()
                return np.argmax(predictions, axis=1)
        else:
            return self.meta_learner.predict(meta_features)

