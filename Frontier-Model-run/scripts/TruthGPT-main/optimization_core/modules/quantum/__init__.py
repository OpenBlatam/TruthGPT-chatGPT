"""
Quantum Computing Systems Package.

Provides lazy-loaded access to three ultra-advanced quantum computing subsystems:

- **Conscious** — Quantum conscious computing
- **Holographic** — Holographic quantum computing
- **Fractal** — Quantum fractal computing

All public symbols are resolved lazily on first access via ``__getattr__``
to keep import time minimal.
"""

from typing import Any

from optimization_core.utils.dependency_manager import resolve_lazy_import

_LAZY_IMPORTS: dict[str, str] = {
    # Conscious
    "QuantumConsciousComputingType": ".conscious",
    "QuantumConsciousOperation": ".conscious",
    "QuantumConsciousLevel": ".conscious",
    "QuantumConsciousConfig": ".conscious",
    "QuantumConsciousMetrics": ".conscious",
    "QuantumConsciousState": ".conscious",
    "UltraAdvancedQuantumConsciousComputingSystem": ".conscious",
    # Holographic
    "HolographicQuantumComputingType": ".holographic",
    "HolographicQuantumOperation": ".holographic",
    "HolographicQuantumComputingLevel": ".holographic",
    "HolographicQuantumComputingConfig": ".holographic",
    "HolographicQuantumComputingMetrics": ".holographic",
    "HolographicQuantumState": ".holographic",
    "UltraAdvancedHolographicQuantumComputingSystem": ".holographic",
    # Fractal
    "QuantumFractalComputingType": ".fractal",
    "QuantumFractalOperation": ".fractal",
    "QuantumFractalComputingLevel": ".fractal",
    "QuantumFractalComputingConfig": ".fractal",
    "QuantumFractalComputingMetrics": ".fractal",
    "QuantumFractalState": ".fractal",
    "UltraAdvancedQuantumFractalComputingSystem": ".fractal",
}


def __getattr__(name: str) -> Any:
    """Resolve public symbols lazily from their respective sub-modules."""
    return resolve_lazy_import(name, __package__ or "quantum", _LAZY_IMPORTS)


__all__ = list(_LAZY_IMPORTS.keys())

