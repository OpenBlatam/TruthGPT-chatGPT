"""
Routing Package
===============
Router implementations for MoE token dispatch.
"""

from .base import BaseRouter, RoutingResult
from .token_router import TokenLevelRouter

# Backward compatibility: legacy imports
try:
    from .pimoe_router import (
        TokenLevelRouter as PiMoETokenLevelRouter,
        PiMoEExpert,
        PiMoESystem,
        create_pimoe_system,
    )
except ImportError:
    pass

__all__ = [
    'BaseRouter',
    'RoutingResult',
    'TokenLevelRouter',
]

