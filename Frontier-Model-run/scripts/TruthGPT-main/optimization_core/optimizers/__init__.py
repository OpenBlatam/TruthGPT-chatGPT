"""
Redirection shim for moved optimizers.
"""
from ..modules.optimizers import *

# Explicitly expose generic_compatibility and core for verify_structure.py
from ..modules.optimizers.compatibility import generic_shims as generic_compatibility
from ..modules.optimizers import core
