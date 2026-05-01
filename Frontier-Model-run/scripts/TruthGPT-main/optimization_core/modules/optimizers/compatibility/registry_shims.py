"""
Compatibility Shims for Optimization Registries
==============================================
Provides backward compatibility for deprecated registry modules.
"""

import warnings
from ..registries import advanced_optimization_registry as aor
from ..registries import advanced_optimization_registry_v2 as aor_v2

def _warn_moved(old_path: str, new_path: str):
    warnings.warn(
        f"{old_path} is deprecated. Please use {new_path} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Redirections for advanced_optimization_registry
class AdvancedOptimizationConfigShim:
    def __getattr__(self, name):
        _warn_moved("optimizers.advanced_optimization_registry", "optimizers.registries.advanced_optimization_registry")
        return getattr(aor, name)

# Simplified approach: export the functions from the module directly in the shim file if needed,
# or just have the shim file import them.
# For consolidation, we'll use classes or module-level __getattr__ if python 3.7+

# To maintain strict compatibility with 'from optimizers.advanced_optimization_registry import X',
# we still need those root files. 
# WAIT - if I remove the root files, imports will FAIL unless I add them to __path__ or something complex.
# The standard way to "clean up" shims is to keep the files but make them as thin as possible.
# My previous plan to delete them might be premature if they are part of the public API.

# HOWEVER, if I want to "clean up" the root, I can move the logic to compatibility 
# and have the root files be 1-line imports from compatibility.

# Let's stick to making the root files as thin as possible.

