"""
Adaptive Learning Enums
======================

Enums for learning modes and states in adaptive systems.
"""
from enum import Enum

class LearningMode(Enum):
    """Learning modes for adaptive systems"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    META_LEARNING = "meta_learning"
    SELF_IMPROVEMENT = "self_improvement"

