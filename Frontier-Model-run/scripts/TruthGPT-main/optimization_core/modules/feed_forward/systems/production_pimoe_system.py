"""
Production-Ready PiMoE System  (backward-compatibility shim)

This module re-exports components from the refactored production system
to maintain backward compatibility with existing imports.
"""

from .refactored_production_system import (
    RefactoredProductionPiMoESystem as ProductionPiMoESystem,
    create_refactored_production_system as create_production_pimoe_system,
    ProductionLogger,
    ProductionMonitor,
    ProductionErrorHandler,
    ProductionRequestQueue,
)

from ..core.refactored_pimoe_base import (
    ProductionConfig,
    ProductionMode,
    LogLevel
)

# Dummy run_production_demo function for backward compatibility
def run_production_demo():
    print("Please use run_refactored_production_demo() instead.")

__all__ = [
    "ProductionMode",
    "LogLevel",
    "ProductionConfig",
    "ProductionLogger",
    "ProductionMonitor",
    "ProductionErrorHandler",
    "ProductionRequestQueue",
    "ProductionPiMoESystem",
    "create_production_pimoe_system",
    "run_production_demo",
]

if __name__ == "__main__":
    run_production_demo()

