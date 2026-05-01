"""
Example demonstrating modular inference with caching and batching.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.base.core_system.core.service_registry import ServiceContainer
from modules.base.core_system.core.services import InferenceService, ModelService
from modules.base.core_system.core.config import ConfigManager

from inference.text_generator import TextGenerator


def setup_inference(model_name: str = "gpt2", use_cache: bool = True):
    """
    Setup inference with modular architecture.
    
    Args:
        model_name: Model name or path
        use_cache: Enable caching
    """
    # 1. Create service container
    container = ServiceContainer()
    
    # 2. Setup model service
    model_service = ModelService()
    model_service.initialize()
    
    # 3. Load model
    model = model_service.load_model(
        model_name=model_name,
        config={
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
    )
    
    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 5. Setup inference service
    inference_service = InferenceService()
    inference_service.configure(
        model=model,
        tokenizer=tokenizer,
        config={
            "use_cache": use_cache,
            "cache_dir": "cache/inference" if use_cache else None,
            "max_batch_size": 8,
            "max_seq_length": 512,
        }
    )
    
    return inference_service


def inference_example():
    """Run inference examples."""
    # Setup
    inference_service = setup_inference("gpt2", use_cache=True)
    
    # Single generation
    print("=" * 60)
    print("Single Generation")
    print("=" * 60)
    result = inference_service.generate("The future of artificial intelligence", {
        "max_new_tokens": 64,
        "temperature": 0.8,
    })
    print(f"Generated: {result}\n")
    
    # Batch generation
    print("=" * 60)
    print("Batch Generation")
    print("=" * 60)
    prompts = [
        "The theory of relativity states that",
        "In machine learning, transformers are",
        "Deep learning has revolutionized",
    ]
    results = inference_service.generate(prompts, {
        "max_new_tokens": 50,
        "temperature": 0.7,
    })
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Generated: {result}\n")
    
    # Profiling
    print("=" * 60)
    print("Performance Profiling")
    print("=" * 60)
    metrics = inference_service.profile(
        "Test prompt for profiling",
        num_runs=10,
    )
    print(f"Average time: {metrics['avg_time']:.4f}s")
    print(f"Throughput: {metrics['throughput']:.2f} samples/sec\n")
    
    # Cache statistics
    if inference_service.text_generator:
        stats = inference_service.get_cache_stats()
        if stats:
            print("=" * 60)
            print("Cache Statistics")
            print("=" * 60)
            print(f"Memory cache size: {stats['memory_cache_size']}")
            print(f"Disk cache size: {stats['disk_cache_size']}")


if __name__ == "__main__":
    inference_example()



