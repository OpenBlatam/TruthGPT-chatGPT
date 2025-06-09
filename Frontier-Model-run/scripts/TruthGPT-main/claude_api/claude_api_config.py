"""
Configuration for Claude API integration
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ClaudeAPIConfig:
    """Configuration for Claude API client."""
    
    api_key: Optional[str] = None
    model_name: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    
    use_optimization_core: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    enable_streaming: bool = False
    enable_function_calling: bool = True
    
    safety_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.safety_settings is None:
            self.safety_settings = {
                'block_harmful_content': True,
                'content_filter_level': 'medium',
                'enable_constitutional_ai': True
            }

def get_default_claude_api_config() -> ClaudeAPIConfig:
    """Get default configuration for Claude API."""
    return ClaudeAPIConfig()

def get_optimized_claude_api_config() -> ClaudeAPIConfig:
    """Get optimized configuration for Claude API with enhanced settings."""
    return ClaudeAPIConfig(
        model_name="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=0.5,
        top_p=0.95,
        use_optimization_core=True,
        enable_caching=True,
        cache_size=2000,
        enable_streaming=True,
        enable_function_calling=True,
        safety_settings={
            'block_harmful_content': True,
            'content_filter_level': 'high',
            'enable_constitutional_ai': True,
            'safety_threshold': 0.95
        }
    )
