"""
Experience Replay Buffer
========================

Buffer for storing and sampling experiences for RL training.
"""
import numpy as np
import logging
from typing import Tuple, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

class ExperienceReplay:
    """Experience replay buffer with prioritization support."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_increment = 0.001
        
        logger.info(f"✅ Experience Replay initialized (capacity: {capacity})")
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, td_error: float = None):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        # Calculate priority
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = 1.0
        
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[Optional[List], Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample batch from buffer using prioritized experience replay."""
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities for sampled experiences based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                priority = (abs(td_error) + 1e-6) ** self.alpha
                self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)

