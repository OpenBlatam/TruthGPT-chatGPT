"""
Basic Usage Examples for TruthGPT Optimization Core
Demonstrates fundamental usage patterns and configurations
"""

import torch
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def basic_transformer_example():
    """
    Basic example of creating and using a transformer model.
    
    This example demonstrates:
    - Creating a transformer model
    - Forward pass
    - Basic configuration
    """
    logger.info("Running basic transformer example...")
    
    try:
        # Import required modules
        from optimization_core.config import create_transformer_config
        from optimization_core.modules.model import create_transformer_model
        
        # Create configuration
        config = create_transformer_config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            vocab_size=50000,
            max_seq_length=1024,
            dropout=0.1
        )
        
        logger.info(f"Created configuration: {config.name}")
        
        # Create model
        model = create_transformer_model(**config.to_dict())
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create sample input
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
        
        logger.info(f"Input shape: {input_ids.shape}")
        logger.info(f"Output logits shape: {outputs['logits'].shape}")
        logger.info(f"Hidden states shape: {outputs['hidden_states'].shape}")
        
        return {
            'model': model,
            'config': config,
            'outputs': outputs
        }
        
    except Exception as e:
        logger.error(f"Basic transformer example failed: {e}")
        raise

def configuration_example():
    """
    Example of using the configuration system.
    
    This example demonstrates:
    - Loading configuration from YAML
    - Environment variable configuration
    - Configuration validation
    """
    logger.info("Running configuration example...")
    
    try:
        from optimization_core.config import ConfigManager, create_transformer_config
        
        # Create configuration manager
        config_manager = ConfigManager()
        
        # Load configuration from YAML file
        try:
            yaml_config = config_manager.load_config_from_file(
                "config/optimization_config.yaml",
                "yaml_config"
            )
            logger.info("Loaded YAML configuration successfully")
        except FileNotFoundError:
            logger.warning("YAML config file not found, using defaults")
            yaml_config = {}
        
        # Load configuration from environment variables
        env_config = config_manager.load_config_from_env(
            prefix="TRUTHGPT_",
            config_name="env_config"
        )
        logger.info("Loaded environment configuration")
        
        # Create transformer configuration
        transformer_config = create_transformer_config(
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            vocab_size=50000
        )
        
        # Validate configuration
        try:
            config_manager.validate_config(transformer_config.to_dict(), "transformer")
            logger.info("Configuration validation passed")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
        
        return {
            'yaml_config': yaml_config,
            'env_config': env_config,
            'transformer_config': transformer_config
        }
        
    except Exception as e:
        logger.error(f"Configuration example failed: {e}")
        raise

def attention_example():
    """
    Example of using different attention mechanisms.
    
    This example demonstrates:
    - Standard multi-head attention
    - Flash attention
    - Attention weight visualization
    """
    logger.info("Running attention example...")
    
    try:
        from optimization_core.modules.attention import (
            create_multi_head_attention,
            create_flash_attention
        )
        
        # Model parameters
        d_model = 512
        n_heads = 8
        seq_len = 128
        batch_size = 2
        
        # Create standard attention
        standard_attention = create_multi_head_attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.1
        )
        
        # Create Flash attention
        flash_attention = create_flash_attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.1,
            use_flash_attention=True
        )
        
        # Create sample input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test standard attention
        logger.info("Testing standard attention...")
        standard_output, standard_weights = standard_attention(x, x, x)
        logger.info(f"Standard attention output shape: {standard_output.shape}")
        logger.info(f"Standard attention weights shape: {standard_weights.shape}")
        
        # Test Flash attention
        logger.info("Testing Flash attention...")
        flash_output, flash_weights = flash_attention(x, x, x)
        logger.info(f"Flash attention output shape: {flash_output.shape}")
        logger.info(f"Flash attention weights shape: {flash_weights.shape}")
        
        # Compare outputs
        output_diff = torch.abs(standard_output - flash_output).mean()
        logger.info(f"Output difference (mean absolute): {output_diff:.6f}")
        
        return {
            'standard_attention': standard_attention,
            'flash_attention': flash_attention,
            'standard_output': standard_output,
            'flash_output': flash_output,
            'output_diff': output_diff
        }
        
    except Exception as e:
        logger.error(f"Attention example failed: {e}")
        raise

def feed_forward_example():
    """
    Example of using different feed-forward networks.
    
    This example demonstrates:
    - Standard feed-forward
    - SwiGLU activation
    - Gated feed-forward
    """
    logger.info("Running feed-forward example...")
    
    try:
        from optimization_core.modules.feed_forward import (
            create_feed_forward,
            create_swiglu
        )
        
        # Model parameters
        d_model = 512
        d_ff = 2048
        seq_len = 128
        batch_size = 2
        
        # Create standard feed-forward
        standard_ffn = create_feed_forward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=0.1,
            architecture="standard"
        )
        
        # Create SwiGLU feed-forward
        swiglu_ffn = create_swiglu(
            d_model=d_model,
            d_ff=d_ff,
            dropout=0.1
        )
        
        # Create sample input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test standard feed-forward
        logger.info("Testing standard feed-forward...")
        standard_output = standard_ffn(x)
        logger.info(f"Standard FFN output shape: {standard_output.shape}")
        
        # Test SwiGLU feed-forward
        logger.info("Testing SwiGLU feed-forward...")
        swiglu_output = swiglu_ffn(x)
        logger.info(f"SwiGLU FFN output shape: {swiglu_output.shape}")
        
        return {
            'standard_ffn': standard_ffn,
            'swiglu_ffn': swiglu_ffn,
            'standard_output': standard_output,
            'swiglu_output': swiglu_output
        }
        
    except Exception as e:
        logger.error(f"Feed-forward example failed: {e}")
        raise

def positional_encoding_example():
    """
    Example of using different positional encodings.
    
    This example demonstrates:
    - Sinusoidal positional encoding
    - Learned positional encoding
    - Rotary embeddings
    """
    logger.info("Running positional encoding example...")
    
    try:
        from optimization_core.modules.embeddings import (
            create_positional_encoding,
            create_rotary_embedding
        )
        
        # Model parameters
        d_model = 512
        seq_len = 128
        batch_size = 2
        
        # Create sinusoidal positional encoding
        sinusoidal_pe = create_positional_encoding(
            encoding_type="sinusoidal",
            d_model=d_model,
            max_length=1024
        )
        
        # Create learned positional encoding
        learned_pe = create_positional_encoding(
            encoding_type="learned",
            d_model=d_model,
            max_length=1024
        )
        
        # Create rotary embedding
        rotary_pe = create_rotary_embedding(
            dim=d_model,
            max_position_embeddings=1024
        )
        
        # Create sample input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test sinusoidal positional encoding
        logger.info("Testing sinusoidal positional encoding...")
        sinusoidal_output = sinusoidal_pe(x)
        logger.info(f"Sinusoidal PE output shape: {sinusoidal_output.shape}")
        
        # Test learned positional encoding
        logger.info("Testing learned positional encoding...")
        learned_output = learned_pe(x)
        logger.info(f"Learned PE output shape: {learned_output.shape}")
        
        # Test rotary embedding
        logger.info("Testing rotary embedding...")
        cos, sin = rotary_pe(x, seq_len)
        logger.info(f"Rotary embedding cos shape: {cos.shape}")
        logger.info(f"Rotary embedding sin shape: {sin.shape}")
        
        return {
            'sinusoidal_pe': sinusoidal_pe,
            'learned_pe': learned_pe,
            'rotary_pe': rotary_pe,
            'sinusoidal_output': sinusoidal_output,
            'learned_output': learned_output,
            'cos': cos,
            'sin': sin
        }
        
    except Exception as e:
        logger.error(f"Positional encoding example failed: {e}")
        raise

def run_all_basic_examples():
    """Run all basic examples."""
    logger.info("Running all basic examples...")
    
    examples = [
        ("Basic Transformer", basic_transformer_example),
        ("Configuration", configuration_example),
        ("Attention", attention_example),
        ("Feed-Forward", feed_forward_example),
        ("Positional Encoding", positional_encoding_example)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            logger.info(f"Running {name} example...")
            result = example_func()
            results[name] = result
            logger.info(f"✓ {name} example completed successfully")
        except Exception as e:
            logger.error(f"✗ {name} example failed: {e}")
            results[name] = None
    
    return results

if __name__ == "__main__":
    # Run all basic examples
    results = run_all_basic_examples()
    
    # Print summary
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"BASIC EXAMPLES SUMMARY")
    print(f"{'='*50}")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total - successful}/{total}")
    print(f"{'='*50}")





