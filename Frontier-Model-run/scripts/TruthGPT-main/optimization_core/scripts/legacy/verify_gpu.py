"""
Verification script for the GPU Accelerator package.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from optimization_core.modules.acceleration.gpu import (
    GPUAcceleratorConfig,
    create_gpu_accelerator,
    create_enhanced_gpu_accelerator,
    gpu_accelerator_context,
    example_gpu_acceleration
)

def verify():
    print("Verifying GPU Accelerator Package Modularization...\n")
    
    print("1. Testing Configuration...")
    config = GPUAcceleratorConfig(device_id=0, enable_memory_pool=True)
    assert config.enable_memory_pool is True
    print("  [OK] Configuration is valid.")
    
    print("2. Testing Basic Accelerator Initialization...")
    accelerator = create_gpu_accelerator(config)
    stats = accelerator.get_stats()
    assert 'device' in stats
    assert 'memory' in stats
    print("  [OK] Basic GPU Accelerator initialized successfully.")
    
    print("3. Testing Enhanced Accelerator Initialization...")
    enhanced = create_enhanced_gpu_accelerator(config)
    enhanced_stats = enhanced.get_enhanced_stats()
    assert 'current_metrics' in enhanced_stats
    print("  [OK] Enhanced GPU Accelerator initialized successfully.")
    
    print("\nAll GPU Accelerator verifications passed! [DONE]")

if __name__ == "__main__":
    verify()

