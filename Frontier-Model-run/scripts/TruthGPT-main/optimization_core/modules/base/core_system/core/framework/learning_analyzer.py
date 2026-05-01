"""
Learning Analyzer Module
Analyzes learning experiences and updates exploration rates
"""

import time
from typing import List, Dict
import numpy as np
import logging

from .config import (
    DEFAULT_BATCH_SIZE_FOR_LEARNING,
    DEFAULT_SUCCESS_THRESHOLD,
    DEFAULT_SUCCESS_RATE_THRESHOLD,
    DEFAULT_EXPLORATION_RATE_MIN,
    DEFAULT_EXPLORATION_RATE_MAX,
    DEFAULT_EXPLORATION_ADJUSTMENT,
    DEFAULT_EXPLORATION_INCREASE
)

logger = logging.getLogger(__name__)


class LearningAnalyzer:
    """Analyzes learning experiences and calculates metrics."""
    
    @staticmethod
    def calculate_success_rate(experiences: List[Dict]) -> float:
        """Calculate success rate from experiences."""
        if not experiences:
            return 0.0
        return sum(1 for exp in experiences if exp['success']) / len(experiences)
    
    @staticmethod
    def calculate_avg_memory_reduction(experiences: List[Dict]) -> float:
        """Calculate average memory reduction from experiences."""
        if not experiences:
            return 0.0
        return float(np.mean([exp['memory_reduction'] for exp in experiences]))
    
    @staticmethod
    def create_learning_record(
        success_rate: float,
        avg_memory_reduction: float
    ) -> Dict:
        """Create a learning history record."""
        return {
            'success_rate': success_rate,
            'avg_memory_reduction': avg_memory_reduction,
            'timestamp': time.time()
        }
    
    @staticmethod
    def adjust_exploration_rate(
        current_rate: float,
        success_rate: float
    ) -> float:
        """Adjust exploration rate based on success rate."""
        if success_rate > DEFAULT_SUCCESS_RATE_THRESHOLD:
            new_rate = current_rate * DEFAULT_EXPLORATION_ADJUSTMENT
        else:
            new_rate = current_rate * DEFAULT_EXPLORATION_INCREASE
        
        return max(
            DEFAULT_EXPLORATION_RATE_MIN,
            min(DEFAULT_EXPLORATION_RATE_MAX, new_rate)
        )







