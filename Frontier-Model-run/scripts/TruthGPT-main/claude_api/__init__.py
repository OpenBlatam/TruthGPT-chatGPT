"""
Claude API Integration for TruthGPT
Provides native Claude API access with optimization_core integration
"""

from .claude_api_client import ClaudeAPIClient, create_claude_api_model, create_claud_api_model
from .claude_api_config import ClaudeAPIConfig
from .claude_api_optimizer import ClaudeAPIOptimizer

__all__ = [
    'ClaudeAPIClient',
    'create_claude_api_model',
    'create_claud_api_model',
    'ClaudeAPIConfig',
    'ClaudeAPIOptimizer'
]
