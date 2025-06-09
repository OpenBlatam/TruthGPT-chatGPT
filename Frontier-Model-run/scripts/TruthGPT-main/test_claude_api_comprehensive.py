"""
Comprehensive test for Claude API integration with optimization_core.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_claude_api_with_optimizations():
    """Test Claude API model with comprehensive optimization_core integration."""
    try:
        from claude_api import create_claud_api_model
        
        config = {
            'model_name': 'claude-3-5-sonnet-20241022',
            'use_optimization_core': True,
            'enable_caching': True,
            'max_tokens': 1000,
            'temperature': 0.7,
            'top_p': 0.9
        }
        
        model = create_claud_api_model(config)
        
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Claude API model instantiated successfully with {param_count:,} parameters")
        
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.shape[0] == 2
        assert output.shape[1] == 16
        logger.info(f"✓ Claude API forward pass successful: {output.shape}")
        
        response = model.generate_text("Test prompt for Claude API optimization")
        assert isinstance(response, str)
        logger.info(f"✓ Claude API text generation working: {response[:50]}...")
        
        stats = model.get_stats()
        assert 'optimization_core_enabled' in stats
        logger.info(f"✓ Claude API optimization stats: {stats}")
        
        from claude_api.claude_api_optimizer import ClaudeAPIOptimizer
        optimizer = ClaudeAPIOptimizer({
            'enable_request_batching': True,
            'enable_response_caching': True,
            'use_fp16': True
        })
        
        optimized_model = optimizer.optimize_model(model)
        assert optimized_model is not None
        logger.info(f"✓ Claude API optimizer integration working")
        
        return True
        
    except Exception as e:
        logger.error(f"Claude API comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_claude_api_batch_optimization():
    """Test Claude API batch optimization features."""
    try:
        from claude_api.claude_api_optimizer import ClaudeAPIOptimizer
        
        optimizer = ClaudeAPIOptimizer({
            'enable_request_batching': True,
            'max_batch_size': 5,
            'concurrent_requests': 2
        })
        
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "How does deep learning work?",
            "What are neural networks?",
            "Describe natural language processing"
        ]
        
        batch_result = optimizer.optimize_batch_requests(prompts)
        
        assert 'optimized_prompts' in batch_result
        assert 'batch_config' in batch_result
        assert 'optimization_stats' in batch_result
        
        logger.info(f"✓ Claude API batch optimization working")
        logger.info(f"  - Batch size: {batch_result['batch_config']['batch_size']}")
        logger.info(f"  - Token savings: {batch_result['optimization_stats']['estimated_token_savings']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Claude API batch optimization test failed: {e}")
        return False

def main():
    """Run comprehensive Claude API tests."""
    logger.info("🧪 Starting comprehensive Claude API tests...")
    
    claude_result = test_claude_api_with_optimizations()
    
    batch_result = test_claude_api_batch_optimization()
    
    logger.info(f"\n📊 Claude API Test Results:")
    logger.info(f"Claude API optimization: {'✓' if claude_result else '✗'}")
    logger.info(f"Batch optimization: {'✓' if batch_result else '✗'}")
    
    if claude_result and batch_result:
        logger.info("🎉 All Claude API tests passed!")
        return True
    else:
        logger.warning("⚠️ Some Claude API tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
