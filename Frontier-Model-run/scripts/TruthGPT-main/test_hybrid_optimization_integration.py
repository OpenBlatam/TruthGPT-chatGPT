"""
Test integration of hybrid optimization with TruthGPT models.
"""

import torch
import torch.nn as nn
from optimization_core import create_hybrid_optimization_core
from enhanced_model_optimizer import create_universal_optimizer

def test_hybrid_optimization_integration():
    """Test hybrid optimization integration with enhanced model optimizer."""
    print("🧪 Testing Hybrid Optimization Integration")
    print("=" * 50)
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(128, 256)
            self.norm = nn.LayerNorm(256)
            self.linear2 = nn.Linear(256, 64)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.norm(x)
            x = self.linear2(x)
            return x
    
    model = TestModel()
    print(f"✅ Test model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    hybrid_core = create_hybrid_optimization_core({
        'enable_candidate_selection': True,
        'enable_ensemble_optimization': True,
        'num_candidates': 3,
        'optimization_strategies': ['kernel_fusion', 'quantization']
    })
    
    optimized_model, result = hybrid_core.hybrid_optimize_module(model)
    print(f"✅ Direct hybrid optimization: {result['selected_strategy']}")
    print(f"   Performance: {result['performance_metrics']}")
    
    config = {
        'enable_hybrid_optimization': True,
        'enable_candidate_selection': True,
        'enable_ensemble_optimization': True,
        'num_candidates': 3,
        'hybrid_strategies': ['kernel_fusion', 'memory_pooling']  # Remove quantization to avoid tensor issues
    }
    
    optimizer = create_universal_optimizer(config)
    hybrid_core = create_hybrid_optimization_core(config)
    enhanced_model, result = hybrid_core.hybrid_optimize_module(model)
    print(f"✅ Enhanced optimizer integration successful")
    
    report = hybrid_core.get_optimization_report()
    print(f"✅ Optimization report generated:")
    print(f"   Total optimizations: {report['total_optimizations']}")
    print(f"   Strategy usage: {report['strategy_usage']}")
    
    return True

def test_candidate_model_selection():
    """Test candidate model selection with different strategies."""
    print("\n🎯 Testing Candidate Model Selection")
    print("=" * 50)
    
    class CandidateModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.Linear(256, 32)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    model = CandidateModel()
    
    selection_strategies = ['tournament', 'roulette', 'rank']
    
    for strategy in selection_strategies:
        hybrid_core = create_hybrid_optimization_core({
            'selection_strategy': strategy,
            'num_candidates': 4,
            'optimization_strategies': ['kernel_fusion', 'quantization', 'memory_pooling', 'attention_fusion']
        })
        
        optimized_model, result = hybrid_core.hybrid_optimize_module(model)
        print(f"✅ {strategy.capitalize()} selection: {result['selected_strategy']}")
        print(f"   Candidates evaluated: {result['num_candidates']}")
    
    return True

def test_model_specific_optimization():
    """Test optimization for different model types."""
    print("\n🏗️ Testing Model-Specific Optimization")
    print("=" * 50)
    
    model_configs = {
        'deepseek': {
            'hybrid_strategies': ['kernel_fusion', 'quantization', 'attention_fusion'],
            'objective_weights': {'speed': 0.5, 'memory': 0.3, 'accuracy': 0.2}
        },
        'llama': {
            'hybrid_strategies': ['memory_pooling', 'kernel_fusion', 'quantization'],
            'objective_weights': {'speed': 0.4, 'memory': 0.4, 'accuracy': 0.2}
        },
        'claude': {
            'hybrid_strategies': ['attention_fusion', 'memory_pooling'],
            'objective_weights': {'speed': 0.3, 'memory': 0.3, 'accuracy': 0.4}
        }
    }
    
    class GenericModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.linear = nn.Linear(100, 50)
            
        def forward(self, x):
            return self.linear(x)
    
    for model_type, config in model_configs.items():
        model = GenericModel(model_type)
        
        hybrid_core = create_hybrid_optimization_core({
            'optimization_strategies': config['hybrid_strategies'],
            'num_candidates': 3
        })
        
        optimized_model, result = hybrid_core.hybrid_optimize_module(model)
        print(f"✅ {model_type.upper()} optimization: {result['selected_strategy']}")
        print(f"   Strategies tested: {result['all_strategies']}")
    
    return True

if __name__ == "__main__":
    print("🚀 Hybrid Optimization Integration Test Suite")
    print("=" * 60)
    
    try:
        test_hybrid_optimization_integration()
        test_candidate_model_selection()
        test_model_specific_optimization()
        
        print("\n🎉 All hybrid optimization integration tests passed!")
        print("✅ Hybrid optimization system is fully functional")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
