"""
Adaptive Learning System
======================

Main orchestrator for adaptive learning and self-improvement mechanisms.
"""
import time
import logging
import pickle
import numpy as np
import random
import torch.nn as nn
from typing import Dict, Any, List
from collections import deque

from .config import AdaptiveLearningConfig
from .enums import LearningMode
from .tracker import PerformanceTracker
from .meta import MetaLearner
from .engine import SelfImprovementEngine

logger = logging.getLogger(__name__)

class AdaptiveLearningSystem:
    """Main adaptive learning system"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.performance_tracker = PerformanceTracker(config)
        self.meta_learner = MetaLearner(config)
        self.self_improvement = SelfImprovementEngine(config)
        self.learning_mode = LearningMode.EXPLORATION
        self.adaptation_history = []
        
        logger.info("✅ Adaptive Learning System initialized")
    
    def adapt(self, model: nn.Module, performance_metrics: Dict[str, float]) -> nn.Module:
        """Adapt model based on performance metrics"""
        logger.info("🧠 Starting adaptive learning...")
        
        # Record performance metrics
        for metric_name, value in performance_metrics.items():
            self.performance_tracker.record_metric(metric_name, value)
        
        # Extract task features for meta-learning
        task_features = self._extract_task_features(model, performance_metrics)
        
        # Learn from this task
        overall_performance = np.mean(list(performance_metrics.values()))
        self.meta_learner.learn_from_task(task_features, overall_performance)
        
        # Evaluate performance for self-improvement
        evaluation = self.self_improvement.evaluate_performance(overall_performance)
        
        # Apply improvements if needed
        if evaluation['improvement_needed']:
            strategy = evaluation['suggested_strategy']
            model = self.self_improvement.apply_improvement(strategy, model)
            
            logger.info(f"🔧 Applied improvement: {strategy}")
        
        # Update learning mode
        self._update_learning_mode()
        
        # Record adaptation
        self.adaptation_history.append({
            'performance_metrics': performance_metrics,
            'task_features': task_features,
            'evaluation': evaluation,
            'learning_mode': self.learning_mode.value,
            'timestamp': time.time()
        })
        
        logger.info("✅ Adaptive learning completed")
        return model
    
    def _extract_task_features(self, model: nn.Module, metrics: Dict[str, float]) -> np.ndarray:
        """Extract features for meta-learning"""
        features = []
        
        # Model features
        total_params = sum(p.numel() for p in model.parameters())
        features.append(min(total_params / 1e6, 100.0))  # Normalize
        
        layer_count = len(list(model.modules()))
        features.append(min(layer_count / 100, 10.0))  # Normalize
        
        # Performance features
        for metric_name, value in metrics.items():
            features.append(min(value, 10.0))  # Normalize
        
        # Pad or truncate to fixed size
        while len(features) < 64:
            features.append(0.0)
        features = features[:64]
        
        return np.array(features)
    
    def _update_learning_mode(self):
        """Update learning mode based on performance"""
        performance_summary = self.performance_tracker.get_performance_summary()
        
        if performance_summary['overall_trend'] == 'improving':
            # Switch to exploitation when improving
            if random.random() < self.config.exploitation_rate:
                self.learning_mode = LearningMode.EXPLOITATION
        else:
            # Switch to exploration when not improving
            if random.random() < self.config.exploration_rate:
                self.learning_mode = LearningMode.EXPLORATION
        
        logger.debug(f"Learning mode: {self.learning_mode.value}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'meta_learning_insights': self.meta_learner.get_learning_insights(),
            'improvement_statistics': self.self_improvement.get_improvement_statistics(),
            'current_learning_mode': self.learning_mode.value,
            'total_adaptations': len(self.adaptation_history),
            'config': {
                'learning_rate': self.config.learning_rate,
                'adaptation_rate': self.config.adaptation_rate,
                'exploration_rate': self.config.exploration_rate,
                'exploitation_rate': self.config.exploitation_rate
            }
        }
    
    def save_learning_state(self, path: str):
        """Save learning state"""
        state = {
            'performance_tracker': {
                'metrics_history': list(self.performance_tracker.metrics_history),
                'performance_trends': dict(self.performance_tracker.performance_trends)
            },
            'meta_learner': {
                'task_memory': list(self.meta_learner.task_memory),
                'meta_model_state': self.meta_learner.meta_model.state_dict()
            },
            'self_improvement': {
                'improvement_memory': list(self.self_improvement.improvement_memory),
                'best_performance': self.self_improvement.best_performance,
                'improvement_patience_counter': self.self_improvement.improvement_patience_counter
            },
            'adaptation_history': self.adaptation_history,
            'learning_mode': self.learning_mode.value
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"✅ Learning state saved to {path}")
    
    def load_learning_state(self, path: str):
        """Load learning state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Restore performance tracker
        self.performance_tracker.metrics_history = deque(state['performance_tracker']['metrics_history'], maxlen=1000)
        self.performance_tracker.performance_trends = state['performance_tracker']['performance_trends']
        
        # Restore meta-learner
        self.meta_learner.task_memory = deque(state['meta_learner']['task_memory'], maxlen=1000)
        self.meta_learner.meta_model.load_state_dict(state['meta_learner']['meta_model_state'])
        
        # Restore self-improvement
        self.self_improvement.improvement_memory = deque(state['self_improvement']['improvement_memory'], maxlen=self.config.improvement_memory_size)
        self.self_improvement.best_performance = state['self_improvement']['best_performance']
        self.self_improvement.improvement_patience_counter = state['self_improvement']['improvement_patience_counter']
        
        # Restore adaptation history
        self.adaptation_history = state['adaptation_history']
        self.learning_mode = LearningMode(state['learning_mode'])
        
        logger.info(f"✅ Learning state loaded from {path}")

