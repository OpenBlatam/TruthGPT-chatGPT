"""
Examples for inference engines.

Demonstrates usage of vLLM, TensorRT-LLM, and other inference engines.
"""
from pathlib import Path

# Example 1: Using vLLM Engine
def example_vllm_engine():
    """Example of using vLLM engine."""
    from inference.vllm_engine import VLLMEngine
    
    # Create engine
    engine = VLLMEngine(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="float16"
    )
    
    # Generate text
    result = engine.generate(
        "What is machine learning?",
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"Generated: {result}")
    return result


# Example 2: Using TensorRT-LLM Engine
def example_tensorrt_engine():
    """Example of using TensorRT-LLM engine."""
    from inference.tensorrt_llm_engine import TensorRTLLMEngine
    
    # Create engine
    engine = TensorRTLLMEngine(
        model_path="/path/to/model",
        precision="fp16",
        max_batch_size=8,
        max_seq_length=512
    )
    
    # Generate text
    results = engine.generate(
        ["Prompt 1", "Prompt 2"],
        max_new_tokens=64,
        temperature=0.7
    )
    
    print(f"Generated: {results}")
    return results


# Example 3: Using Engine Factory
def example_engine_factory():
    """Example of using engine factory."""
    from inference.engine_factory import create_inference_engine, EngineType
    
    # Auto-select best engine
    engine = create_inference_engine(
        model="mistral-7b",
        engine_type=EngineType.AUTO
    )
    
    # Or specify engine
    engine = create_inference_engine(
        model="mistral-7b",
        engine_type=EngineType.VLLM
    )
    
    result = engine.generate("Hello, world!")
    return result


# Example 4: Using Base Engine Interface
def example_base_engine():
    """Example of using base engine interface."""
    from inference.base_engine import BaseInferenceEngine, GenerationConfig
    
    # All engines implement the same interface
    engines = [
        create_inference_engine("model1", engine_type=EngineType.VLLM),
        create_inference_engine("model2", engine_type=EngineType.TENSORRT_LLM),
    ]
    
    # Use them interchangeably
    for engine in engines:
        result = engine.generate("test prompt")
        print(f"Result: {result}")


# Example 5: Using Decorators
def example_with_decorators():
    """Example of using decorators."""
    from inference.utils.decorators import (
        log_execution_time,
        retry_on_failure,
        cache_result
    )
    
    @log_execution_time("generation")
    @retry_on_failure(max_attempts=3)
    @cache_result(ttl=3600)
    def generate_with_retry(engine, prompt):
        return engine.generate(prompt)
    
    result = generate_with_retry(engine, "test")
    return result













