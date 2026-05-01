"""
Multi-Task Learning Enums
=========================

Enums for task types, relationships, and sharing strategies.
"""
from enum import Enum

class TaskType(Enum):
    """Task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

class TaskRelationship(Enum):
    """Task relationships"""
    INDEPENDENT = "independent"
    RELATED = "related"
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DEPENDENT = "dependent"

class SharingStrategy(Enum):
    """Sharing strategies"""
    HARD_SHARING = "hard_sharing"
    SOFT_SHARING = "soft_sharing"
    TASK_SPECIFIC = "task_specific"
    ADAPTIVE_SHARING = "adaptive_sharing"
    HIERARCHICAL_SHARING = "hierarchical_sharing"

