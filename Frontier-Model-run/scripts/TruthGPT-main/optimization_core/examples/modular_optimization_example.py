"""
Modular Optimization Example
Demonstrating unified modular optimization using UnifiedOptimizer
"""

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any

from optimization_core.modules.optimizers import UnifiedOptimizer, OptimizationLevel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_model():
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

def demonstrate_modular_levels():
    """Demonstrate different optimization levels in the unified system."""
    print("🔧 Modular Optimization Levels")
    print("=" * 60)
    
    model = create_simple_model()
    
    levels = [
        OptimizationLevel.BASIC,
        OptimizationLevel.ADVANCED,
        OptimizationLevel.EXPERT,
        OptimizationLevel.MASTER,
        OptimizationLevel.LEGENDARY,
        OptimizationLevel.DIVINE
    ]
    
    for level in levels:
        print(f"\n🔧 Testing {level.value.upper()} optimization...")
        
        config = {
            "use_mixed_precision": True if level != OptimizationLevel.BASIC else False,
            "use_torch_compile": True if level in [OptimizationLevel.EXPERT, OptimizationLevel.MASTER, OptimizationLevel.LEGENDARY, OptimizationLevel.DIVINE] else False,
            "compile_mode": "max-autotune" if level == OptimizationLevel.DIVINE else "default"
        }
        
        optimizer = UnifiedOptimizer(config=config, level=level)
        
        start_time = time.time()
        result = optimizer.optimize(model)
        duration = time.time() - start_time
        
        print(f"    ⚡ Speed improvement: {result.speed_improvement:.2f}x (Estimated)")
        print(f"    🛠️  Techniques: {', '.join(result.techniques_applied)}")
        print(f"    ⏱️  Optimization time: {duration*1000:.2f}ms")

def main():
    print("🔧 Modular Optimization Demonstration (Unified)")
    print("=" * 70)
    
    try:
        demonstrate_modular_levels()
        print("\n✅ All modular examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Modular example failed: {e}")

if __name__ == "__main__":
    main()

