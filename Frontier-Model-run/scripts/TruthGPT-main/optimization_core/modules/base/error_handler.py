"""
Error handling shim for modules.base.
"""
from .core_system.core.exceptions import *

class ErrorConfig:
    pass

class ErrorHandler:
    pass

def handle_errors(func):
    return func

__all__ = ['ErrorHandler', 'ErrorConfig', 'handle_errors']
