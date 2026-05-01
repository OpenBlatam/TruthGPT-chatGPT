"""
Test script for vr_optimization
"""
import sys
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from optimization_core.modules.vr_optimization import (
    UltraVROptimizationEngine,
    VROptimizationConfig,
    VROptimizationLevel,
    ImmersiveTechnology
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_vr():
    logger.info("🔍 Verifying VR Optimization Package...")
    
    try:
        config = VROptimizationConfig(
            level=VROptimizationLevel.VR_ADVANCED,
            immersive_technologies=[
                ImmersiveTechnology.VIRTUAL_REALITY,
                ImmersiveTechnology.NEURAL_INTERFACE
            ]
        )
        engine = UltraVROptimizationEngine(config)
        
        # Test optimization
        class DummySystem:
            def add_capability(self, cap):
                logger.info(f"Capability added: {cap}")
                
        system = DummySystem()
        result = engine.optimize_system(system)
        
        assert result.success is True
        assert "immersive_rendering" in engine.rendering_engine.create_engine()["type"]
        
        logger.info("✅ VR Optimization Package verified successfully!")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify_vr()

