"""
Ultra-fast modular interface components
Following deep learning best practices
"""

from .gradio_interface import FastGradioInterface, GradioConfig, create_interface
from .api_interface import FastAPIInterface, APIConfig, create_api
from .cli_interface import CLIInterface, CLIConfig, create_cli
from .web_interface import WebInterface, WebConfig, create_web_interface
from .stream_interface import StreamInterface, StreamConfig, create_stream_interface

__all__ = [
    # Gradio interface
    'FastGradioInterface', 'GradioConfig', 'create_interface',
    
    # API interface
    'FastAPIInterface', 'APIConfig', 'create_api',
    
    # CLI interface
    'CLIInterface', 'CLIConfig', 'create_cli',
    
    # Web interface
    'WebInterface', 'WebConfig', 'create_web_interface',
    
    # Stream interface
    'StreamInterface', 'StreamConfig', 'create_stream_interface'
]



