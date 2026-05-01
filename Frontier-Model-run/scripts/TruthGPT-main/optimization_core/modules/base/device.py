"""
Device management shim for modules.base.
"""
from ..acceleration.gpu.device import (
    GPUDeviceManager as DeviceManager,
    GPUAcceleratorConfig as DeviceConfig,
)

def get_optimal_device():
    """Proxy for get_optimal_device."""
    from ..acceleration.gpu.device import get_optimal_device as _get
    return _get()

__all__ = ['DeviceManager', 'DeviceConfig', 'get_optimal_device']
