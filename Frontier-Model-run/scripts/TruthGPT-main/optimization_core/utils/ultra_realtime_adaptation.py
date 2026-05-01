"""
Ultra-Advanced Real-Time Adaptation System Module
================================================

This module provides real-time adaptation capabilities for TruthGPT models,
including dynamic model updates, online learning, and adaptive optimization.

Author: TruthGPT Ultra-Advanced Optimization Core Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pydantic import BaseModel
from enum import Enum
import json
import pickle
from pathlib import Path
import concurrent.futures
from collections import defaultdict, deque
import math
import statistics
import warnings
import threading
import queue
import asyncio

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class AdaptationMode(Enum):
    """Adaptation modes for real-time learning."""
    CONTINUOUS = "continuous"
    BATCH = "batch"
    STREAMING = "streaming"
    EVENT_DRIVEN = "event_driven"
    HYBRID = "hybrid"

class AdaptationStrategy(Enum):
    """Adaptation strategies."""
    GRADIENT_DESCENT = "gradient_descent"
    META_LEARNING = "meta_learning"
    ONLINE_LEARNING = "online_learning"
    TRANSFER_LEARNING = "transfer_learning"
    CONTINUAL_LEARNING = "continual_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"

class AdaptationTrigger(Enum):
    """Adaptation triggers."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    THRESHOLD_BASED = "threshold_based"

class LearningRateSchedule(Enum):
    """Learning rate schedules."""
    CONSTANT = "constant"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_ANNEALING = "cosine_annealing"
    ADAPTIVE = "adaptive"
    PERFORMANCE_BASED = "performance_based"

class AdaptationConfig(BaseModel):
    """Configuration for real-time adaptation."""
    adaptation_mode: AdaptationMode = AdaptationMode.CONTINUOUS
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.ONLINE_LEARNING
    adaptation_trigger: AdaptationTrigger = AdaptationTrigger.PERFORMANCE_DEGRADATION
    learning_rate_schedule: LearningRateSchedule = LearningRateSchedule.ADAPTIVE
    initial_learning_rate: float = 0.001
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 0.01
    adaptation_threshold: float = 0.1
    adaptation_frequency: int = 100
    memory_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 10
    performance_window: int = 100
    drift_detection_window: int = 1000
    device: str = "auto"
    log_level: str = "INFO"
    output_dir: str = "./adaptation_results"

class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_window)
        self.loss_history = deque(maxlen=config.performance_window)
        self.accuracy_history = deque(maxlen=config.performance_window)
        self.current_performance = 0.0
        self.baseline_performance = 0.0
        self.performance_trend = 0.0
        
    def update_performance(self, loss: float, accuracy: float):
        """Update performance metrics."""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        
        # Calculate composite performance score
        self.current_performance = accuracy - loss
        
        # Update performance history
        self.performance_history.append(self.current_performance)
        
        # Calculate performance trend
        if len(self.performance_history) >= 10:
            recent_performance = list(self.performance_history)[-10:]
            self.performance_trend = statistics.mean(recent_performance) - statistics.mean(list(self.performance_history)[:-10])
        
        # Update baseline if needed
        if self.baseline_performance == 0.0:
            self.baseline_performance = self.current_performance
    
    def should_adapt(self) -> bool:
        """Determine if adaptation is needed."""
        if len(self.performance_history) < 10:
            return False
        
        # Check performance degradation
        current_avg = statistics.mean(list(self.performance_history)[-10:])
        baseline_avg = self.baseline_performance
        
        performance_degradation = (baseline_avg - current_avg) / baseline_avg
        
        return performance_degradation > self.config.adaptation_threshold
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'current_performance': self.current_performance,
            'baseline_performance': self.baseline_performance,
            'performance_trend': self.performance_trend,
            'average_loss': statistics.mean(self.loss_history) if self.loss_history else 0.0,
            'average_accuracy': statistics.mean(self.accuracy_history) if self.accuracy_history else 0.0,
            'performance_degradation': (self.baseline_performance - self.current_performance) / self.baseline_performance if self.baseline_performance > 0 else 0.0
        }

class DriftDetector:
    """Data and concept drift detection."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.data_history = deque(maxlen=config.drift_detection_window)
        self.concept_history = deque(maxlen=config.drift_detection_window)
        self.drift_threshold = 0.1
        self.drift_detected = False
        
    def detect_data_drift(self, new_data: torch.Tensor) -> bool:
        """Detect data drift."""
        if len(self.data_history) < 10:
            self.data_history.append(new_data)
            return False
        
        # Calculate statistical distance between new data and historical data
        historical_data = torch.cat(list(self.data_history))
        
        # Use KL divergence or other distance metrics
        drift_score = self._calculate_drift_score(new_data, historical_data)
        
        self.data_history.append(new_data)
        
        drift_detected = drift_score > self.drift_threshold
        
        if drift_detected:
            logger.warning(f"Data drift detected with score: {drift_score:.4f}")
        
        return drift_detected
    
    def detect_concept_drift(self, predictions: torch.Tensor, targets: torch.Tensor) -> bool:
        """Detect concept drift."""
        if len(self.concept_history) < 10:
            self.concept_history.append((predictions, targets))
            return False
        
        # Calculate prediction accuracy trend
        current_accuracy = (predictions.argmax(dim=1) == targets).float().mean().item()
        
        historical_accuracies = []
        for hist_pred, hist_target in self.concept_history:
            hist_accuracy = (hist_pred.argmax(dim=1) == hist_target).float().mean().item()
            historical_accuracies.append(hist_accuracy)
        
        avg_historical_accuracy = statistics.mean(historical_accuracies)
        
        # Detect significant accuracy drop
        accuracy_drop = avg_historical_accuracy - current_accuracy
        
        self.concept_history.append((predictions, targets))
        
        drift_detected = accuracy_drop > self.drift_threshold
        
        if drift_detected:
            logger.warning(f"Concept drift detected with accuracy drop: {accuracy_drop:.4f}")
        
        return drift_detected
    
    def _calculate_drift_score(self, new_data: torch.Tensor, historical_data: torch.Tensor) -> float:
        """Calculate drift score between new and historical data."""
        # Simplified drift detection using mean and std differences
        new_mean = new_data.mean().item()
        new_std = new_data.std().item()
        
        hist_mean = historical_data.mean().item()
        hist_std = historical_data.std().item()
        
        # Calculate normalized difference
        mean_diff = abs(new_mean - hist_mean) / (hist_std + 1e-8)
        std_diff = abs(new_std - hist_std) / (hist_std + 1e-8)
        
        drift_score = (mean_diff + std_diff) / 2.0
        
        return drift_score

class AdaptiveLearningRate:
    """Adaptive learning rate scheduler."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.current_lr = config.initial_learning_rate
        self.performance_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        
    def update_learning_rate(self, performance: float, loss: float):
        """Update learning rate based on performance."""
        self.performance_history.append(performance)
        self.lr_history.append(self.current_lr)
        
        if self.config.learning_rate_schedule == LearningRateSchedule.ADAPTIVE:
            self._adaptive_update(performance, loss)
        elif self.config.learning_rate_schedule == LearningRateSchedule.PERFORMANCE_BASED:
            self._performance_based_update(performance)
        elif self.config.learning_rate_schedule == LearningRateSchedule.EXPONENTIAL_DECAY:
            self._exponential_decay_update()
        elif self.config.learning_rate_schedule == LearningRateSchedule.COSINE_ANNEALING:
            self._cosine_annealing_update()
        else:  # CONSTANT
            pass  # Keep current learning rate
    
    def _adaptive_update(self, performance: float, loss: float):
        """Adaptive learning rate update."""
        if len(self.performance_history) < 5:
            return
        
        # Calculate performance trend
        recent_performance = list(self.performance_history)[-5:]
        avg_recent_performance = statistics.mean(recent_performance)
        
        if len(self.performance_history) >= 10:
            older_performance = list(self.performance_history)[-10:-5]
            avg_older_performance = statistics.mean(older_performance)
            
            if avg_recent_performance > avg_older_performance:
                # Performance improving, increase learning rate slightly
                self.current_lr *= 1.01
            else:
                # Performance degrading, decrease learning rate
                self.current_lr *= 0.99
        
        # Clamp learning rate
        self.current_lr = max(self.config.min_learning_rate, 
                             min(self.config.max_learning_rate, self.current_lr))
    
    def _performance_based_update(self, performance: float):
        """Performance-based learning rate update."""
        if performance > 0.8:  # High performance
            self.current_lr *= 0.95  # Decrease learning rate
        elif performance < 0.5:  # Low performance
            self.current_lr *= 1.05  # Increase learning rate
        
        # Clamp learning rate
        self.current_lr = max(self.config.min_learning_rate, 
                             min(self.config.max_learning_rate, self.current_lr))
    
    def _exponential_decay_update(self):
        """Exponential decay learning rate update."""
        decay_rate = 0.95
        self.current_lr *= decay_rate
        
        # Clamp learning rate
        self.current_lr = max(self.config.min_learning_rate, self.current_lr)
    
    def _cosine_annealing_update(self):
        """Cosine annealing learning rate update."""
        if len(self.lr_history) < 2:
            return
        
        # Simple cosine annealing
        max_lr = self.config.max_learning_rate
        min_lr = self.config.min_learning_rate
        
        # Calculate cosine annealing
        step = len(self.lr_history)
        cosine_lr = min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * step / 100)) / 2
        
        self.current_lr = cosine_lr

class OnlineLearner:
    """Online learning implementation."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.memory_buffer = deque(maxlen=config.memory_size)
        self.update_counter = 0
        self.last_update_time = time.time()
        
    def add_sample(self, input_data: torch.Tensor, target: torch.Tensor):
        """Add sample to memory buffer."""
        self.memory_buffer.append((input_data, target))
    
    def should_update(self) -> bool:
        """Check if model should be updated."""
        return len(self.memory_buffer) >= self.config.update_frequency
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get batch from memory buffer."""
        if len(self.memory_buffer) < self.config.batch_size:
            batch_size = len(self.memory_buffer)
        else:
            batch_size = self.config.batch_size
        
        # Random sampling from memory buffer
        batch_indices = random.sample(range(len(self.memory_buffer)), batch_size)
        
        batch_inputs = []
        batch_targets = []
        
        for idx in batch_indices:
            input_data, target = self.memory_buffer[idx]
            batch_inputs.append(input_data)
            batch_targets.append(target)
        
        return torch.stack(batch_inputs), torch.stack(batch_targets)
    
    def update_model(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                    criterion: nn.Module, learning_rate: float) -> Dict[str, float]:
        """Update model with online learning."""
        if not self.should_update():
            return {'loss': 0.0, 'accuracy': 0.0}
        
        # Get batch
        batch_inputs, batch_targets = self.get_batch()
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == batch_targets).float().mean().item()
        
        self.update_counter += 1
        self.last_update_time = time.time()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'learning_rate': learning_rate
        }

class MetaLearner:
    """Meta-learning for rapid adaptation."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.meta_optimizer = None
        self.meta_learning_rate = 0.001
        self.few_shot_examples = deque(maxlen=100)
        
    def add_few_shot_example(self, input_data: torch.Tensor, target: torch.Tensor):
        """Add few-shot learning example."""
        self.few_shot_examples.append((input_data, target))
    
    def meta_update(self, model: nn.Module, support_set: List[Tuple[torch.Tensor, torch.Tensor]], 
                   query_set: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Perform meta-learning update."""
        if len(support_set) < 2 or len(query_set) < 1:
            return {'meta_loss': 0.0, 'meta_accuracy': 0.0}
        
        # Create a copy of the model for meta-update
        meta_model = copy.deepcopy(model)
        meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=self.meta_learning_rate)
        
        # Inner loop: adapt to support set
        for _ in range(5):  # Few gradient steps
            meta_optimizer.zero_grad()
            
            support_loss = 0.0
            for support_input, support_target in support_set:
                support_output = meta_model(support_input)
                support_loss += F.cross_entropy(support_output, support_target)
            
            support_loss /= len(support_set)
            support_loss.backward()
            meta_optimizer.step()
        
        # Outer loop: evaluate on query set
        query_loss = 0.0
        query_accuracy = 0.0
        
        with torch.no_grad():
            for query_input, query_target in query_set:
                query_output = meta_model(query_input)
                query_loss += F.cross_entropy(query_output, query_target)
                
                predictions = query_output.argmax(dim=1)
                query_accuracy += (predictions == query_target).float().mean().item()
        
        query_loss /= len(query_set)
        query_accuracy /= len(query_set)
        
        return {
            'meta_loss': query_loss.item(),
            'meta_accuracy': query_accuracy
        }

class RealTimeAdaptationManager:
    """Main manager for real-time adaptation."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(config)
        self.drift_detector = DriftDetector(config)
        self.adaptive_lr = AdaptiveLearningRate(config)
        self.online_learner = OnlineLearner(config)
        self.meta_learner = MetaLearner(config)
        
        # Adaptation state
        self.adaptation_count = 0
        self.last_adaptation_time = time.time()
        self.adaptation_history = deque(maxlen=1000)
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def set_model(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Set model and optimizer for adaptation."""
        self.model = model
        self.optimizer = optimizer
        logger.info("Model and optimizer set for real-time adaptation")
    
    def process_sample(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """Process a single sample for real-time adaptation."""
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_data)
            loss = self.criterion(output, target)
            predictions = output.argmax(dim=1)
            accuracy = (predictions == target).float().mean().item()
        
        # Update performance monitoring
        self.performance_monitor.update_performance(loss.item(), accuracy)
        
        # Detect drift
        data_drift = self.drift_detector.detect_data_drift(input_data)
        concept_drift = self.drift_detector.detect_concept_drift(output, target)
        
        # Add sample to online learner
        self.online_learner.add_sample(input_data, target)
        
        # Check if adaptation is needed
        adaptation_needed = self._should_adapt(data_drift, concept_drift)
        
        adaptation_result = None
        if adaptation_needed:
            adaptation_result = self._perform_adaptation(input_data, target)
        
        # Update learning rate
        self.adaptive_lr.update_learning_rate(accuracy, loss.item())
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'data_drift': data_drift,
            'concept_drift': concept_drift,
            'adaptation_needed': adaptation_needed,
            'adaptation_result': adaptation_result,
            'learning_rate': self.adaptive_lr.current_lr,
            'performance_metrics': self.performance_monitor.get_performance_metrics()
        }
    
    def _should_adapt(self, data_drift: bool, concept_drift: bool) -> bool:
        """Determine if adaptation should be performed."""
        if self.config.adaptation_trigger == AdaptationTrigger.PERFORMANCE_DEGRADATION:
            return self.performance_monitor.should_adapt()
        elif self.config.adaptation_trigger == AdaptationTrigger.DATA_DRIFT:
            return data_drift
        elif self.config.adaptation_trigger == AdaptationTrigger.CONCEPT_DRIFT:
            return concept_drift
        elif self.config.adaptation_trigger == AdaptationTrigger.THRESHOLD_BASED:
            return (self.adaptation_count % self.config.adaptation_frequency == 0)
        elif self.config.adaptation_trigger == AdaptationTrigger.SCHEDULED:
            return (time.time() - self.last_adaptation_time) > 60  # 1 minute
        else:  # MANUAL
            return False
    
    def _perform_adaptation(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """Perform model adaptation."""
        logger.info(f"Performing adaptation #{self.adaptation_count + 1}")
        
        start_time = time.time()
        
        adaptation_result = {}
        
        if self.config.adaptation_strategy == AdaptationStrategy.ONLINE_LEARNING:
            adaptation_result = self._online_learning_adaptation()
        elif self.config.adaptation_strategy == AdaptationStrategy.META_LEARNING:
            adaptation_result = self._meta_learning_adaptation(input_data, target)
        elif self.config.adaptation_strategy == AdaptationStrategy.GRADIENT_DESCENT:
            adaptation_result = self._gradient_descent_adaptation(input_data, target)
        elif self.config.adaptation_strategy == AdaptationStrategy.FEW_SHOT_LEARNING:
            adaptation_result = self._few_shot_adaptation(input_data, target)
        else:  # CONTINUAL_LEARNING
            adaptation_result = self._continual_learning_adaptation()
        
        adaptation_time = time.time() - start_time
        
        # Update adaptation state
        self.adaptation_count += 1
        self.last_adaptation_time = time.time()
        
        # Record adaptation
        self.adaptation_history.append({
            'adaptation_count': self.adaptation_count,
            'adaptation_time': adaptation_time,
            'strategy': self.config.adaptation_strategy.value,
            'result': adaptation_result,
            'timestamp': time.time()
        })
        
        logger.info(f"Adaptation completed in {adaptation_time:.4f}s")
        
        return {
            'adaptation_count': self.adaptation_count,
            'adaptation_time': adaptation_time,
            'strategy': self.config.adaptation_strategy.value,
            'result': adaptation_result
        }
    
    def _online_learning_adaptation(self) -> Dict[str, float]:
        """Perform online learning adaptation."""
        return self.online_learner.update_model(
            self.model, self.optimizer, self.criterion, self.adaptive_lr.current_lr
        )
    
    def _meta_learning_adaptation(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Perform meta-learning adaptation."""
        # Add current sample to few-shot examples
        self.meta_learner.add_few_shot_example(input_data, target)
        
        # Get recent examples for support and query sets
        recent_examples = list(self.meta_learner.few_shot_examples)[-10:]
        
        if len(recent_examples) < 3:
            return {'meta_loss': 0.0, 'meta_accuracy': 0.0}
        
        # Split into support and query sets
        support_set = recent_examples[:-2]
        query_set = recent_examples[-2:]
        
        return self.meta_learner.meta_update(self.model, support_set, query_set)
    
    def _gradient_descent_adaptation(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Perform gradient descent adaptation."""
        self.optimizer.zero_grad()
        
        output = self.model(input_data)
        loss = self.criterion(output, target)
        
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            predictions = output.argmax(dim=1)
            accuracy = (predictions == target).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'learning_rate': self.adaptive_lr.current_lr
        }
    
    def _few_shot_adaptation(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Perform few-shot learning adaptation."""
        # Similar to meta-learning but with fewer examples
        self.meta_learner.add_few_shot_example(input_data, target)
        
        recent_examples = list(self.meta_learner.few_shot_examples)[-5:]
        
        if len(recent_examples) < 2:
            return {'few_shot_loss': 0.0, 'few_shot_accuracy': 0.0}
        
        # Use all examples as support set
        support_set = recent_examples
        query_set = recent_examples[-1:]  # Use last example as query
        
        return self.meta_learner.meta_update(self.model, support_set, query_set)
    
    def _continual_learning_adaptation(self) -> Dict[str, float]:
        """Perform continual learning adaptation."""
        # Prevent catastrophic forgetting by using experience replay
        if len(self.online_learner.memory_buffer) < self.config.batch_size:
            return {'continual_loss': 0.0, 'continual_accuracy': 0.0}
        
        # Sample from memory buffer
        batch_inputs, batch_targets = self.online_learner.get_batch()
        
        self.optimizer.zero_grad()
        
        output = self.model(batch_inputs)
        loss = self.criterion(output, batch_targets)
        
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            predictions = output.argmax(dim=1)
            accuracy = (predictions == batch_targets).float().mean().item()
        
        return {
            'continual_loss': loss.item(),
            'continual_accuracy': accuracy,
            'learning_rate': self.adaptive_lr.current_lr
        }
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        return {
            'adaptation_count': self.adaptation_count,
            'last_adaptation_time': self.last_adaptation_time,
            'adaptation_history_size': len(self.adaptation_history),
            'performance_metrics': self.performance_monitor.get_performance_metrics(),
            'memory_buffer_size': len(self.online_learner.memory_buffer),
            'few_shot_examples': len(self.meta_learner.few_shot_examples),
            'current_learning_rate': self.adaptive_lr.current_lr,
            'config': {
                'adaptation_mode': self.config.adaptation_mode.value,
                'adaptation_strategy': self.config.adaptation_strategy.value,
                'adaptation_trigger': self.config.adaptation_trigger.value,
                'learning_rate_schedule': self.config.learning_rate_schedule.value
            }
        }
    
    def reset_adaptation(self):
        """Reset adaptation state."""
        self.adaptation_count = 0
        self.last_adaptation_time = time.time()
        self.adaptation_history.clear()
        self.performance_monitor.performance_history.clear()
        self.drift_detector.data_history.clear()
        self.drift_detector.concept_history.clear()
        self.online_learner.memory_buffer.clear()
        self.meta_learner.few_shot_examples.clear()
        
        logger.info("Adaptation state reset")

# Factory functions
def create_adaptation_config(adaptation_mode: AdaptationMode = AdaptationMode.CONTINUOUS,
                           adaptation_strategy: AdaptationStrategy = AdaptationStrategy.ONLINE_LEARNING,
                           adaptation_trigger: AdaptationTrigger = AdaptationTrigger.PERFORMANCE_DEGRADATION,
                           **kwargs) -> AdaptationConfig:
    """Create adaptation configuration."""
    return AdaptationConfig(
        adaptation_mode=adaptation_mode,
        adaptation_strategy=adaptation_strategy,
        adaptation_trigger=adaptation_trigger,
        **kwargs
    )

def create_performance_monitor(config: AdaptationConfig) -> PerformanceMonitor:
    """Create performance monitor."""
    return PerformanceMonitor(config)

def create_drift_detector(config: AdaptationConfig) -> DriftDetector:
    """Create drift detector."""
    return DriftDetector(config)

def create_adaptive_learning_rate(config: AdaptationConfig) -> AdaptiveLearningRate:
    """Create adaptive learning rate scheduler."""
    return AdaptiveLearningRate(config)

def create_online_learner(config: AdaptationConfig) -> OnlineLearner:
    """Create online learner."""
    return OnlineLearner(config)

def create_meta_learner(config: AdaptationConfig) -> MetaLearner:
    """Create meta learner."""
    return MetaLearner(config)

def create_real_time_adaptation_manager(config: Optional[AdaptationConfig] = None) -> RealTimeAdaptationManager:
    """Create real-time adaptation manager."""
    if config is None:
        config = create_adaptation_config()
    return RealTimeAdaptationManager(config)

# Example usage
def example_real_time_adaptation():
    """Example of real-time adaptation."""
    # Create configuration
    config = create_adaptation_config(
        adaptation_mode=AdaptationMode.CONTINUOUS,
        adaptation_strategy=AdaptationStrategy.ONLINE_LEARNING,
        adaptation_trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
        adaptation_threshold=0.05,
        update_frequency=5
    )
    
    # Create adaptation manager
    adaptation_manager = create_real_time_adaptation_manager(config)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Set model and optimizer
    adaptation_manager.set_model(model, optimizer)
    
    # Simulate real-time data stream
    for i in range(100):
        # Generate sample data
        input_data = torch.randn(1, 10)
        target = torch.randint(0, 2, (1,))
        
        # Process sample
        result = adaptation_manager.process_sample(input_data, target)
        
        if i % 20 == 0:
            print(f"Sample {i}: Loss={result['loss']:.4f}, Accuracy={result['accuracy']:.4f}, "
                  f"Adaptation={result['adaptation_needed']}, LR={result['learning_rate']:.6f}")
    
    # Get adaptation statistics
    stats = adaptation_manager.get_adaptation_statistics()
    print(f"Adaptation statistics: {stats}")
    
    return stats

if __name__ == "__main__":
    # Run example
    example_real_time_adaptation()

