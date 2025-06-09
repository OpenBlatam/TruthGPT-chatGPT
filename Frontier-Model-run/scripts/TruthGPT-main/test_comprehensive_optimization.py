"""
Comprehensive test suite for optimization_core integration across all TruthGPT models.
"""

import torch
import torch.nn as nn
# import pytest  # Not available, using basic testing
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_claude_api_optimization():
    """Test Claude API model with optimization_core integration."""
    try:
        from claude_api import create_claud_api_model
        
        config = {
            'model_name': 'claude-3-5-sonnet-20241022',
            'use_optimization_core': True,
            'enable_caching': True,
            'max_tokens': 1000
        }
        
        model = create_claud_api_model(config)
        
        assert model is not None
        logger.info(f"✓ Claude API model instantiated successfully")
        
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.shape[0] == 2
        assert output.shape[1] == 16
        logger.info(f"✓ Claude API forward pass successful: {output.shape}")
        
        response = model.generate_text("Test prompt for Claude API")
        assert isinstance(response, str)
        logger.info(f"✓ Claude API text generation working")
        
        stats = model.get_stats()
        assert 'optimization_core_enabled' in stats
        logger.info(f"✓ Claude API optimization stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Claude API optimization test failed: {e}")
        return False

def test_enhanced_model_optimizer():
    """Test enhanced model optimizer with comprehensive layer replacement."""
    try:
        from enhanced_model_optimizer import create_universal_optimizer
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_norm = nn.LayerNorm(512)
                self.linear1 = nn.Linear(512, 2048)
                self.linear2 = nn.Linear(2048, 512)
                
            def forward(self, x):
                x = self.layer_norm(x)
                x = self.linear1(x)
                x = self.linear2(x)
                return x
        
        model = TestModel()
        
        optimizer = create_universal_optimizer({
            'enable_fp16': True,
            'enable_gradient_checkpointing': True,
            'use_advanced_normalization': True,
            'use_enhanced_mlp': True,
            'use_mcts_optimization': True
        })
        
        optimized_model = optimizer.optimize_model(model, "test_model")
        
        input_tensor = torch.randn(2, 512)
        with torch.no_grad():
            output = optimized_model(input_tensor)
        
        assert output.shape == (2, 512)
        logger.info(f"✓ Enhanced model optimizer working: {output.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced model optimizer test failed: {e}")
        return False

def test_all_model_optimizations():
    """Test optimization_core integration across all TruthGPT models."""
    models_to_test = [
        ('deepseek_v3', 'Frontier-Model-run.models.deepseek_v3', 'create_deepseek_v3_model'),
        ('llama_3_1_405b', 'Frontier-Model-run.models.llama_3_1_405b', 'create_llama_3_1_405b_model'),
        ('claude_3_5_sonnet', 'Frontier-Model-run.models.claude_3_5_sonnet', 'create_claude_3_5_sonnet_model'),
        ('viral_clipper', 'variant.viral_clipper', 'create_viral_clipper_model'),
        ('brand_analyzer', 'brandkit.brand_analyzer', 'create_brand_analyzer_model'),
        ('qwen_model', 'qwen_variant.qwen_model', 'create_qwen_model'),
        ('claud_api', 'claude_api.claude_api_client', 'create_claud_api_model')
    ]
    
    results = {}
    
    for model_name, module_path, factory_func in models_to_test:
        try:
            module = __import__(module_path, fromlist=[factory_func])
            create_model = getattr(module, factory_func)
            
            config = {
                'hidden_size': 512,
                'num_layers': 2,
                'num_attention_heads': 8,
                'vocab_size': 1000
            }
            
            model = create_model(config)
            
            input_ids = torch.randint(0, 1000, (2, 16))
            with torch.no_grad():
                output = model(input_ids)
            
            results[model_name] = {
                'status': 'success',
                'output_shape': output.shape,
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
            logger.info(f"✓ {model_name} optimization test passed")
            
        except Exception as e:
            results[model_name] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"✗ {model_name} optimization test failed: {e}")
    
    return results

def main():
    """Run comprehensive optimization tests."""
    logger.info("🧪 Starting comprehensive optimization tests...")
    
    claude_result = test_claude_api_optimization()
    
    optimizer_result = test_enhanced_model_optimizer()
    
    all_models_results = test_all_model_optimizations()
    
    total_models = len(all_models_results)
    successful_models = sum(1 for r in all_models_results.values() if r['status'] == 'success')
    
    logger.info(f"\n📊 Comprehensive Optimization Test Results:")
    logger.info(f"Claude API optimization: {'✓' if claude_result else '✗'}")
    logger.info(f"Enhanced model optimizer: {'✓' if optimizer_result else '✗'}")
    logger.info(f"Model optimizations: {successful_models}/{total_models} successful")
    
    for model_name, result in all_models_results.items():
        if result['status'] == 'success':
            logger.info(f"  ✓ {model_name}: {result['parameters']:,} parameters")
        else:
            logger.info(f"  ✗ {model_name}: {result['error']}")
    
    if claude_result and optimizer_result and successful_models == total_models:
        logger.info("🎉 All comprehensive optimization tests passed!")
        return True
    else:
        logger.warning("⚠️ Some optimization tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
