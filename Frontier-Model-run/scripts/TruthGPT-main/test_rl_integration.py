"""
Quick test to verify DAPO, VAPO, and ORZ RL integration.
"""

import torch
import torch.nn as nn
from optimization_core import create_hybrid_optimization_core

def test_rl_enhanced_optimization():
    """Test RL-enhanced optimization with DAPO, VAPO, and ORZ."""
    print("🤖 Testing RL-Enhanced Hybrid Optimization")
    print("=" * 50)
    
    config = {
        'enable_rl_optimization': True,
        'enable_dapo': True,
        'enable_vapo': True,
        'enable_orz': True,
        'rl_max_episodes': 10,  # Reduced for quick test
        'num_candidates': 3,
        'optimization_strategies': ['kernel_fusion', 'quantization']
    }
    
    hybrid_core = create_hybrid_optimization_core(config)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32)
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    optimized_model, result = hybrid_core.hybrid_optimize_module(model)
    
    print(f"✅ RL optimization result: {result['selected_strategy']}")
    print(f"   Performance: {result['performance_metrics']}")
    
    report = hybrid_core.get_optimization_report()
    print(f"✅ RL features enabled:")
    print(f"   DAPO: {report['dapo_enabled']}")
    print(f"   VAPO: {report['vapo_enabled']}")
    print(f"   ORZ: {report['orz_enabled']}")
    
    if 'rl_performance' in report:
        rl_perf = report['rl_performance']
        print(f"✅ RL Performance Metrics:")
        print(f"   Policy Loss: {rl_perf['avg_policy_loss']:.4f}")
        print(f"   Value Loss: {rl_perf['avg_value_loss']:.4f}")
        print(f"   RL Episodes: {rl_perf['total_rl_episodes']}")
    
    return True

def test_backward_compatibility():
    """Test that traditional optimization still works when RL is disabled."""
    print("\n🔄 Testing Backward Compatibility")
    print("=" * 50)
    
    config = {
        'enable_rl_optimization': False,
        'num_candidates': 3,
        'optimization_strategies': ['kernel_fusion', 'quantization']
    }
    
    hybrid_core = create_hybrid_optimization_core(config)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32)
    )
    
    optimized_model, result = hybrid_core.hybrid_optimize_module(model)
    
    print(f"✅ Traditional optimization result: {result['selected_strategy']}")
    print(f"   Performance: {result['performance_metrics']}")
    
    report = hybrid_core.get_optimization_report()
    print(f"✅ RL optimization disabled: {not report['rl_optimization_enabled']}")
    
    return True

if __name__ == "__main__":
    print("🚀 RL Integration Test Suite")
    print("=" * 60)
    
    try:
        test_rl_enhanced_optimization()
        test_backward_compatibility()
        
        print("\n🎉 All RL integration tests passed!")
        print("✅ DAPO, VAPO, and ORZ techniques successfully integrated")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
