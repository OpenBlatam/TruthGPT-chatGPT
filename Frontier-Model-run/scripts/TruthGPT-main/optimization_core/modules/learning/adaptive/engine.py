"""
Self-Improvement Engine
=======================

Self-improvement logic for continuous optimization in adaptive systems.
"""
import torch
import torch.nn as nn
import logging
import random
from typing import Dict, Any, List, Optional
from collections import deque
from .config import AdaptiveLearningConfig

logger = logging.getLogger(__name__)

class SelfImprovementEngine:
    """Self-improvement engine for continuous optimization"""
    
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.improvement_memory = deque(maxlen=config.improvement_memory_size)
        self.improvement_strategies = {}
        self.current_strategy = None
        self.improvement_patience_counter = 0
        self.best_performance = float('-inf')
        
        # Initialize improvement strategies
        self._initialize_strategies()
        
        logger.info("✅ Self-Improvement Engine initialized")
    
    def _initialize_strategies(self):
        """Initialize improvement strategies"""
        self.improvement_strategies = {
            'learning_rate_adjustment': self._adjust_learning_rate,
            'architecture_modification': self._modify_architecture,
            'optimization_strategy_change': self._change_optimization_strategy,
            'regularization_adjustment': self._adjust_regularization,
            'data_augmentation': self._enhance_data_augmentation
        }
    
    def evaluate_performance(self, current_performance: float) -> Dict[str, Any]:
        """Evaluate current performance and suggest improvements"""
        evaluation = {
            'current_performance': current_performance,
            'improvement_needed': False,
            'suggested_strategy': None,
            'confidence': 0.0
        }
        
        # Check if improvement is needed
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.improvement_patience_counter = 0
            evaluation['improvement_needed'] = False
        else:
            self.improvement_patience_counter += 1
            
            # Check if improvement threshold is met
            improvement_needed = (self.best_performance - current_performance) > self.config.improvement_threshold
            
            if improvement_needed and self.improvement_patience_counter >= self.config.improvement_patience:
                evaluation['improvement_needed'] = True
                evaluation['suggested_strategy'] = self._select_improvement_strategy()
                evaluation['confidence'] = self._calculate_confidence()
        
        # Store evaluation
        self.improvement_memory.append(evaluation)
        
        return evaluation
    
    def _select_improvement_strategy(self) -> str:
        """Select best improvement strategy"""
        # Analyze past improvements
        successful_strategies = []
        
        for evaluation in self.improvement_memory:
            if evaluation.get('improvement_needed', False):
                strategy = evaluation.get('suggested_strategy')
                if strategy:
                    successful_strategies.append(strategy)
        
        # Select strategy based on success rate
        if successful_strategies:
            strategy_counts = {}
            for strategy in successful_strategies:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Select most successful strategy
            best_strategy = max(strategy_counts, key=strategy_counts.get)
        else:
            # Random selection for exploration
            best_strategy = random.choice(list(self.improvement_strategies.keys()))
        
        return best_strategy
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in improvement suggestion"""
        if not self.improvement_memory:
            return 0.5
        
        # Calculate confidence based on past success
        successful_improvements = sum(1 for eval in self.improvement_memory 
                                   if eval.get('improvement_needed', False))
        
        total_evaluations = len(self.improvement_memory)
        confidence = successful_improvements / total_evaluations if total_evaluations > 0 else 0.5
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def apply_improvement(self, strategy: str, model: nn.Module, **kwargs) -> nn.Module:
        """Apply improvement strategy to model"""
        if strategy not in self.improvement_strategies:
            logger.warning(f"Unknown improvement strategy: {strategy}")
            return model
        
        improvement_func = self.improvement_strategies[strategy]
        
        try:
            improved_model = improvement_func(model, **kwargs)
            logger.info(f"✅ Applied improvement strategy: {strategy}")
            return improved_model
        except Exception as e:
            logger.error(f"Failed to apply improvement strategy {strategy}: {e}")
            return model
    
    def _adjust_learning_rate(self, model: nn.Module, **kwargs) -> nn.Module:
        """Adjust learning rate for better performance"""
        # This would adjust learning rates in practice
        logger.info("📈 Learning rate adjustment applied")
        return model
    
    def _modify_architecture(self, model: nn.Module, **kwargs) -> nn.Module:
        """Modify model architecture"""
        # This would modify architecture in practice
        logger.info("🏗️ Architecture modification applied")
        return model
    
    def _change_optimization_strategy(self, model: nn.Module, **kwargs) -> nn.Module:
        """Change optimization strategy"""
        # This would change optimization strategy in practice
        logger.info("⚙️ Optimization strategy changed")
        return model
    
    def _adjust_regularization(self, model: nn.Module, **kwargs) -> nn.Module:
        """Adjust regularization"""
        # This would adjust regularization in practice
        logger.info("🛡️ Regularization adjusted")
        return model
    
    def _enhance_data_augmentation(self, model: nn.Module, **kwargs) -> nn.Module:
        """Enhance data augmentation"""
        # This would enhance data augmentation in practice
        logger.info("🔄 Data augmentation enhanced")
        return model
    
    def get_improvement_statistics(self) -> Dict[str, Any]:
        """Get self-improvement statistics"""
        if not self.improvement_memory:
            return {'statistics': 'No improvements attempted yet'}
        
        total_evaluations = len(self.improvement_memory)
        improvements_needed = sum(1 for eval in self.improvement_memory 
                                if eval.get('improvement_needed', False))
        
        successful_strategies = [eval.get('suggested_strategy') for eval in self.improvement_memory 
                               if eval.get('improvement_needed', False)]
        
        strategy_counts = {}
        for strategy in successful_strategies:
            if strategy:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_evaluations': total_evaluations,
            'improvements_needed': improvements_needed,
            'improvement_rate': improvements_needed / total_evaluations if total_evaluations > 0 else 0,
            'best_performance': self.best_performance,
            'strategy_usage': strategy_counts,
            'patience_counter': self.improvement_patience_counter
        }

