#!/usr/bin/env python3
"""
Quick optimization test to verify Claude API integration and core functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Frontier-Model-run'))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_claude_api_quick():
    """Quick test of Claude API integration."""
    try:
        from claude_api.claude_api_client import create_claud_api_model
        
        config = {
            'use_optimization_core': True,
            'enable_caching': True,
            'max_tokens': 100
        }
        
        model = create_claud_api_model(config)
        logger.info(f"✅ Claude API model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
            logger.info(f"✅ Forward pass successful: {output.shape}")
        
        response = model.generate_text("Test prompt")
        logger.info(f"✅ Text generation working: {response[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Claude API test failed: {e}")
        return False

def test_optimization_core_integration():
    """Test optimization_core integration."""
    try:
        from optimization_core import OptimizedLayerNorm, MemoryOptimizations
        
        norm = OptimizedLayerNorm(512)
        x = torch.randn(2, 10, 512)
        output = norm(x)
        logger.info(f"✅ OptimizedLayerNorm working: {output.shape}")
        
        mem_opt = MemoryOptimizations()
        logger.info(f"✅ MemoryOptimizations available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization core test failed: {e}")
        return False

def main():
    logger.info("🚀 Quick Optimization Test Suite")
    logger.info("=" * 40)
    
    results = []
    
    logger.info("Testing Claude API integration...")
    results.append(("Claude API", test_claude_api_quick()))
    
    logger.info("Testing optimization_core integration...")
    results.append(("Optimization Core", test_optimization_core_integration()))
    
    logger.info("\n📊 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Summary: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Ready for push.")
    else:
        logger.info("⚠️ Some tests failed. Check logs above.")

if __name__ == "__main__":
    main()
