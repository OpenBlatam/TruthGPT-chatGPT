"""
Continual Learning Enums
========================

Enums for continual learning strategies, replay methods, and memory types.
"""
from enum import Enum

class CLStrategy(Enum):
    """Continual learning strategies"""
    EWC = "ewc"
    REPLAY_BUFFER = "replay_buffer"
    PROGRESSIVE_NETWORKS = "progressive_networks"
    MULTI_TASK_LEARNING = "multi_task_learning"
    LIFELONG_LEARNING = "lifelong_learning"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    DOMAIN_ADAPTATION = "domain_adaptation"

class ReplayStrategy(Enum):
    """Replay strategies"""
    RANDOM_REPLAY = "random_replay"
    STRATEGIC_REPLAY = "strategic_replay"
    EXPERIENCE_REPLAY = "experience_replay"
    GENERATIVE_REPLAY = "generative_replay"
    PROTOTYPE_REPLAY = "prototype_replay"
    CORE_SET_REPLAY = "core_set_replay"

class MemoryType(Enum):
    """Memory types"""
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    WORKING_MEMORY = "working_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    SHORT_TERM_MEMORY = "short_term_memory"

