"""
Modern TruthGPT Example
Demonstrating deep learning best practices for LLM optimization
"""

import torch
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Import our modern components
from optimization_core.modules.optimizers import UnifiedOptimizer, OptimizationLevel
from optimization_core.training import Trainer, TrainingConfig
try:
    from optimization_core.examples.gradio_interface import TruthGPTGradioInterface
except ImportError:
    TruthGPTGradioInterface = None


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('truthgpt_example.log')
        ]
    )
    return logging.getLogger("TruthGPTExample")


def create_sample_data() -> List[str]:
    """Create sample training data"""
    sample_texts = [
        "Hello, how are you today?",
        "What is the weather like?",
        "Tell me about artificial intelligence.",
        "How does machine learning work?",
        "What are the benefits of deep learning?",
        "Explain neural networks in simple terms.",
        "What is the difference between AI and ML?",
        "How do transformers work?",
        "What is attention mechanism?",
        "Explain the concept of embeddings.",
        "What is fine-tuning in deep learning?",
        "How do you train a language model?",
        "What are the challenges in NLP?",
        "Explain the concept of transfer learning.",
        "What is the future of AI?",
        "How do you evaluate language models?",
        "What is the role of data in AI?",
        "Explain the concept of bias in AI.",
        "What are the ethical considerations in AI?",
        "How do you deploy AI models in production?"
    ]
    
    # Repeat data for more training examples
    return sample_texts * 5


def demonstrate_model_initialization():
    """Demonstrate model initialization with best practices"""
    logger = logging.getLogger("ModelInit")
    logger.info("=== Model Initialization Demo ===")
    
    model_name = "microsoft/DialoGPT-medium"
    
    # Create configuration
    config = {
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True,
        "use_flash_attention": True
    }
    
    logger.info(f"Configuration: {config}")
    
    # Initialize model
    logger.info(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except OSError:
        logger.warning(f"Model {model_name} not found locally or internet issue. Using simplified dummy model for demo.")
        config_obj = AutoConfig.from_pretrained("gpt2")
        config_obj.n_layer = 2 # Tiny model
        model = AutoModelForCausalLM.from_config(config_obj)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Optimize model
    logger.info("Applying Unified Optimization...")
    optimizer = UnifiedOptimizer(config=config, level=OptimizationLevel.ADVANCED)
    result = optimizer.optimize(model)
    model = result.optimized_model
    
    # Get model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(result.performance_metrics.get("device", "cpu"))
    }
    logger.info(f"Model Info: {json.dumps(model_info, indent=2)}")
    
    return model, tokenizer, config


def demonstrate_training_pipeline():
    """Demonstrate training pipeline with best practices"""
    logger = logging.getLogger("TrainingPipeline")
    logger.info("=== Training Pipeline Demo ===")
    
    # Create sample data
    sample_texts = create_sample_data()
    logger.info(f"Created {len(sample_texts)} training examples")
    
    # Initialize model
    model, tokenizer, model_config = demonstrate_model_initialization()
    
    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=3,  # Small for demo
        early_stopping_patience=2,
        eval_interval=50,
        use_wandb=False,  # Disable for demo
        batch_size=4,
        max_grad_norm=1.0,
        use_mixed_precision=True
    )
    
    # Create Trainer
    trainer = Trainer(model, tokenizer, training_config)
    
    try:
        # Prepare data
        logger.info("Preparing data...")
        train_loader, val_loader, test_loader = trainer.prepare_data(sample_texts)
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate model
        logger.info("Evaluating model...")
        eval_metrics = trainer.evaluate(test_loader)
        
        # Create results
        results = {
            'history': history,
            'eval_metrics': eval_metrics,
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Final metrics: {eval_metrics}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        # Don't raise, just return empty for demo robustness
        return {}
    finally:
        trainer.cleanup()


def demonstrate_text_generation():
    """Demonstrate text generation capabilities"""
    logger = logging.getLogger("TextGeneration")
    logger.info("=== Text Generation Demo ===")
    
    # Initialize model
    model, tokenizer, config = demonstrate_model_initialization()
    model.eval() # Ensure eval mode
    
    # Sample prompts
    prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?",
    ]
    
    logger.info("Generating text samples...")
    
    for i, prompt in enumerate(prompts):
        try:
            logger.info(f"Prompt {i+1}: {prompt}")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    temperature=1.0,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Generated: {generated_text}")
            logger.info("-" * 50)
            
        except Exception as e:
            logger.error(f"Generation error for prompt '{prompt}': {e}")


def demonstrate_optimization_techniques():
    """Demonstrate various optimization techniques"""
    logger = logging.getLogger("OptimizationTechniques")
    logger.info("=== Optimization Techniques Demo ===")
    
    try:
        model_name = "microsoft/DialoGPT-medium"
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except:
        logger.info("Skipping technique demo due to model load failure")
        return
    
    # Mixed Precision Training
    logger.info("1. Mixed Precision Training:")
    config_mixed = {"use_mixed_precision": True}
    optimizer = UnifiedOptimizer(config=config_mixed, level=OptimizationLevel.ADVANCED)
    optimizer.optimize(model)
    logger.info(f"Mixed precision configured.")
    
    # Gradient Checkpointing
    logger.info("2. Gradient Checkpointing:")
    config_checkpoint = {"use_gradient_checkpointing": True}
    optimizer = UnifiedOptimizer(config=config_checkpoint, level=OptimizationLevel.BASIC)
    optimizer.optimize(model)
    logger.info(f"Gradient checkpointing configured.")
    
    # Compiler
    logger.info("3. Torch Compile:")
    config_compile = {"use_torch_compile": True}
    optimizer = UnifiedOptimizer(config=config_compile, level=OptimizationLevel.EXPERT)
    logger.info(f"Torch compile configured in Expert level.")


def demonstrate_gradio_interface():
    """Demonstrate Gradio interface (optional)"""
    logger = logging.getLogger("GradioInterface")
    logger.info("=== Gradio Interface Demo ===")
    
    if TruthGPTGradioInterface is None:
        logger.warning("Gradio interface module not found or failed to import.")
        return

    try:
        # Create Gradio interface
        interface = TruthGPTGradioInterface()
        logger.info("Gradio interface created successfully")
        logger.info("To launch the interface, run: interface.launch()")
        
    except Exception as e:
        logger.error(f"Gradio interface error: {e}")


def main():
    """Main demonstration function"""
    logger = setup_logging()
    logger.info("=== TruthGPT Modern Example (Unified Architecture) ===")
    logger.info("Demonstrating unified optimization best practices")
    
    try:
        # 1. Model Initialization
        logger.info("\n" + "="*50)
        demonstrate_model_initialization()
        
        # 2. Training Pipeline
        logger.info("\n" + "="*50)
        demonstrate_training_pipeline()
        
        # 3. Text Generation
        logger.info("\n" + "="*50)
        demonstrate_text_generation()
        
        # 4. Optimization Techniques
        logger.info("\n" + "="*50)
        demonstrate_optimization_techniques()
        
        # 5. Gradio Interface (optional)
        logger.info("\n" + "="*50)
        demonstrate_gradio_interface()
        
        logger.info("\n=== Example completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        # Don't raise in main to avoid ugly tracebacks


if __name__ == "__main__":
    main()

