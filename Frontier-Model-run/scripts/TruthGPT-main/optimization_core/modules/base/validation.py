"""
Validation shim for modules.base.
"""
# This might be tricky as validation is a directory
from .core_system.core.validation import *

class ValidationConfig:
    pass

class InputValidator:
    pass

class OutputValidator:
    pass

__all__ = ['InputValidator', 'OutputValidator', 'ValidationConfig']
