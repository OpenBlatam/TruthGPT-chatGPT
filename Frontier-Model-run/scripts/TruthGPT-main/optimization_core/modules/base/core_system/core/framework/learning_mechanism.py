"""
Learning Mechanism Module
Manages AI learning from optimization experiences
"""

import time
from typing import Dict, List, Optional
from collections import deque
import logging

from .config import (
    DEFAULT_BATCH_SIZE_FOR_LEARNING,
    DEFAULT_SUCCESS_THRESHOLD
)
from .learning_analyzer import LearningAnalyzer

logger = logging.getLogger(__name__)


class LearningMechanism:
    """Manages AI learning from optimization experiences."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        exploration_rate: float = 0.1,
        memory_decay: float = 0.95,
        adaptation_rate: float = 0.1,
        experience_buffer_size: int = 10000,
        learning_history_size: int = 1000
    ):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.memory_decay = memory_decay
        self.adaptation_rate = adaptation_rate
        self.experience_buffer = deque(maxlen=experience_buffer_size)
        self.learning_history = deque(maxlen=learning_history_size)
    
    def record_experience(
        self,
        strategy: str,
        confidence: float,
        memory_reduction: float
    ) -> None:
        """Record an optimization experience."""
        experience = {
            'strategy': strategy,
            'confidence': confidence,
            'memory_reduction': memory_reduction,
            'success': memory_reduction > DEFAULT_SUCCESS_THRESHOLD,
            'timestamp': time.time()
        }
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > DEFAULT_BATCH_SIZE_FOR_LEARNING:
            self._update_learning()
    
    def _update_learning(self) -> None:
        """Update learning based on recent experiences."""
        recent_experiences = list(self.experience_buffer)[-DEFAULT_BATCH_SIZE_FOR_LEARNING:]
        
        success_rate = LearningAnalyzer.calculate_success_rate(recent_experiences)
        avg_memory_reduction = LearningAnalyzer.calculate_avg_memory_reduction(recent_experiences)
        
        learning_record = LearningAnalyzer.create_learning_record(
            success_rate, avg_memory_reduction
        )
        self.learning_history.append(learning_record)
        
        self.exploration_rate = LearningAnalyzer.adjust_exploration_rate(
            self.exploration_rate, success_rate
        )
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.exploration_rate
    
    def get_learning_history(self) -> List[Dict]:
        """Get learning history."""
        return list(self.learning_history)
    
    def get_experience_buffer(self) -> List[Dict]:
        """Get experience buffer."""
        return list(self.experience_buffer)
    
    def restore_state(
        self,
        learning_history: List[Dict],
        experience_buffer: List[Dict],
        exploration_rate: float
    ) -> None:
        """Restore learning mechanism state from saved data."""
        self.exploration_rate = exploration_rate
        self.learning_history = deque(learning_history, maxlen=self.learning_history.maxlen)
        self.experience_buffer = deque(experience_buffer, maxlen=self.experience_buffer.maxlen)


