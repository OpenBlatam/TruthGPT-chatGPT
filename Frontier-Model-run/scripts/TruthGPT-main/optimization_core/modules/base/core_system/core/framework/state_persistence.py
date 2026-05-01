"""
State Persistence for AI Optimizer
"""

import pickle
import logging
from typing import Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

from .config import (
    DEFAULT_EXPLORATION_RATE,
    DEFAULT_EXPERIENCE_BUFFER_SIZE,
    DEFAULT_LEARNING_HISTORY_SIZE
)
from .learning_mechanism import LearningMechanism
from .metrics_calculator import AIOptimizationLevel

logger = logging.getLogger(__name__)


class OptimizerState(BaseModel):
    """Pydantic model defending the strict structural integrity of the saved state."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    neural_network_state: Dict[str, Any] = Field(..., description="PyTorch state_dict for the model")
    optimizer_state: Dict[str, Any] = Field(..., description="PyTorch state_dict for the optimizer")
    learning_history: List[Any] = Field(default_factory=list, description="Historical learning actions")
    experience_buffer: List[Any] = Field(default_factory=list, description="Experience buffer tuples")
    exploration_rate: float = Field(default=DEFAULT_EXPLORATION_RATE, description="Current epsilon/exploration rate")
    optimization_level: int = Field(..., description="Optimization tier level value")


class StatePersistence:
    """Handles saving and loading optimizer state with validation."""
    
    @staticmethod
    def save_state(
        neural_network_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        learning_history: list,
        experience_buffer: list,
        exploration_rate: float,
        optimization_level: AIOptimizationLevel,
        filepath: str
    ) -> None:
        """Save optimizer state to file."""
        
        # SOTA 2025: Strict Validation before writing to disk
        state_model = OptimizerState(
            neural_network_state=neural_network_state,
            optimizer_state=optimizer_state,
            learning_history=learning_history,
            experience_buffer=experience_buffer,
            exploration_rate=exploration_rate,
            optimization_level=optimization_level.value
        )
        
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_model.model_dump(), f)
        
        logger.info(f"💾 AI state saved to {filepath}")
    
    @staticmethod
    def load_state(filepath: str) -> Dict[str, Any]:
        """Load optimizer state from file."""
        try:
            with open(filepath, 'rb') as f:
                raw_state = pickle.load(f)
            
            # Re-validate state on load to ensure structural integrity
            state_model = OptimizerState(**raw_state)
            
            logger.info(f"📁 AI state loaded from {filepath}")
            return state_model.model_dump()
        except Exception as e:
            logger.error(f"Failed to load AI state: {e}")
            raise
    
    @staticmethod
    def restore_learning_mechanism(state: Dict[str, Any]) -> LearningMechanism:
        """Restore learning mechanism from saved state."""
        learning_hist = state.get('learning_history', [])
        exp_buffer = state.get('experience_buffer', [])
        exp_rate = state.get('exploration_rate', DEFAULT_EXPLORATION_RATE)
        
        learning_mechanism = LearningMechanism(
            exploration_rate=exp_rate,
            experience_buffer_size=DEFAULT_EXPERIENCE_BUFFER_SIZE,
            learning_history_size=DEFAULT_LEARNING_HISTORY_SIZE
        )
        
        learning_mechanism.restore_state(
            learning_history=learning_hist,
            experience_buffer=exp_buffer,
            exploration_rate=exp_rate
        )
        
        return learning_mechanism



