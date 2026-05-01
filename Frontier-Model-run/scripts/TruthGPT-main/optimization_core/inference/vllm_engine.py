"""
vLLM Inference Engine - 5-10x faster than PyTorch inference.

This module provides a high-performance inference engine using vLLM,
which implements PagedAttention and continuous batching for optimal throughput.
"""
import logging
from typing import List, Optional, Union, Dict, Any
import os
from pathlib import Path

from .base_engine import BaseInferenceEngine, GenerationConfig
from .utils.validators import (
    validate_generation_params,
    validate_positive_int,
    validate_float_range,
    validate_precision,
    validate_quantization,
    validate_non_empty_string,
)
from .utils.prompt_utils import (
    normalize_prompts,
    handle_single_prompt,
    extract_generated_text,
)

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None
    logger.warning(
        "vLLM not available. Install with: pip install vllm>=0.2.0"
    )


class VLLMEngine(BaseInferenceEngine):
    """
    High-performance inference engine using vLLM.
    
    Features:
    - PagedAttention: 3-5x memory reduction
    - Continuous batching: Optimal GPU utilization
    - Multi-GPU support: Tensor parallelism
    - Quantization: INT8/FP8 support
    """
    
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs
    ):
        """
        Initialize vLLM engine.
        
        Args:
            model: Model name or path (HuggingFace format)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum sequence length
            dtype: Data type (auto, float16, bfloat16, float32)
            quantization: Quantization method (awq, gptq, None)
            trust_remote_code: Trust remote code from HuggingFace
            **kwargs: Additional vLLM arguments
        
        Raises:
            ImportError: If vLLM is not installed
            ValueError: If parameters are invalid
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm>=0.2.0"
            )
        
        # Initialize base class
        super().__init__(model=model, **kwargs)
        
        # Validate parameters using shared validators
        validate_non_empty_string(str(model), "model")
        validate_positive_int(tensor_parallel_size, "tensor_parallel_size")
        validate_float_range(
            gpu_memory_utilization,
            "gpu_memory_utilization",
            0.0, 1.0,
            inclusive_min=False,
            inclusive_max=True
        )
        
        if max_model_len is not None:
            validate_positive_int(max_model_len, "max_model_len")
        
        valid_dtypes = {"auto", "float16", "bfloat16", "float32"}
        validate_precision(dtype, valid_dtypes)
        
        if quantization is not None:
            valid_quantizations = {"awq", "gptq", "squeezellm"}
            validate_quantization(quantization, valid_quantizations)
        
        self.model_name = str(model)
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.quantization = quantization
        
        logger.info(f"Initializing vLLM engine with model: {model}")
        
        try:
            # Build LLM engine
            engine_kwargs = {
                "model": model,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
                "dtype": dtype,
                **kwargs
            }
            
            if max_model_len:
                engine_kwargs["max_model_len"] = max_model_len
            
            if quantization:
                engine_kwargs["quantization"] = quantization
            
            self.llm = self._initialize_engine(**engine_kwargs)
            self._set_initialized(True)
            
            logger.info(
                f"vLLM engine initialized successfully "
                f"(tensor_parallel_size={tensor_parallel_size}, "
                f"dtype={dtype}, quantization={quantization})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}", exc_info=True)
            raise
    
    def _initialize_engine(self, **kwargs) -> Any:
        """Initialize the vLLM engine."""
        return LLM(**kwargs)
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> Union[InferenceResult, List[InferenceResult]]:
        """
        Generate text and return structured InferenceResult.
        """
        import time
        # Validate generation parameters using shared validators
        validate_generation_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        
        # Normalize prompts using shared utilities
        prompts_list, was_single = normalize_prompts(prompts)
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                max_tokens=max_tokens,
                stop=stop,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
            
            # Generate
            start_ts = time.monotonic()
            outputs = self.llm.generate(prompts_list, sampling_params)
            latency_ms = (time.monotonic() - start_ts) * 1000
            
            # Extract generated text and package into InferenceResult
            if not outputs:
                logger.warning("No outputs generated from vLLM")
                res = [InferenceResult(text="", model_name=self.model_name, latency_ms=latency_ms)]
                return handle_single_prompt(res, was_single)
            
            results = []
            for output in outputs:
                results.append(InferenceResult(
                    text=output.outputs[0].text,
                    tokens_generated=len(output.outputs[0].token_ids),
                    model_name=self.model_name,
                    latency_ms=latency_ms / len(outputs),
                    finish_reason=output.outputs[0].finish_reason,
                    metadata={"prompt_tokens": len(output.prompt_token_ids)}
                ))
            
            return handle_single_prompt(results, was_single)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate text: {e}") from e
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Generate text for a batch of prompts (optimized for batching).
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated texts
        """
        return self.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Generate text for a batch of prompts (optimized for batching).
        
        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated texts
        """
        return self.generate(
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )


class AsyncVLLMEngine:
    """
    Async vLLM engine for high-throughput serving.
    
    Use this for production serving with async/await support.
    """
    
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        """
        Initialize async vLLM engine.
        
        Args:
            model: Model name or path
            tensor_parallel_size: Number of GPUs
            **kwargs: Additional engine arguments
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed")
        
        self.model_name = model
        
        # Build async engine args
        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"Async vLLM engine initialized: {model}")
    
    async def generate_async(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        **kwargs
    ) -> InferenceResult:
        """
        Generate text asynchronously and return InferenceResult.
        """
        import time
        # Validate parameters using shared validators
        validate_non_empty_string(prompt, "prompt")
        validate_non_empty_string(request_id, "request_id")
        validate_generation_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95  # Default for async
        )
        
        try:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Submit request
            start_ts = time.monotonic()
            self.engine.add_request(request_id, prompt, sampling_params)
            
            # Wait for result
            async for request_output in self.engine.generate(prompt, sampling_params):
                if request_output.finished:
                    latency_ms = (time.monotonic() - start_ts) * 1000
                    if request_output.outputs and len(request_output.outputs) > 0:
                        out = request_output.outputs[0]
                        return InferenceResult(
                            text=out.text,
                            tokens_generated=len(out.token_ids),
                            model_name=self.model_name,
                            latency_ms=latency_ms,
                            finish_reason=out.finish_reason
                        )
                    else:
                        logger.warning(f"Empty output for request {request_id}")
                        return InferenceResult(text="", model_name=self.model_name, latency_ms=latency_ms)
            
            logger.warning(f"Request {request_id} did not complete")
            return InferenceResult(text="", model_name=self.model_name)
            
        except Exception as e:
            logger.error(f"Async generation failed for request {request_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate text asynchronously: {e}") from e


# Factory function for easy integration
def create_vllm_engine(
    model: str,
    use_async: bool = False,
    **kwargs
) -> Union[VLLMEngine, AsyncVLLMEngine]:
    """
    Factory function to create vLLM engine.
    
    Args:
        model: Model name or path
        use_async: Use async engine
        **kwargs: Engine arguments
    
    Returns:
        VLLM engine instance
    """
    if use_async:
        return AsyncVLLMEngine(model, **kwargs)
    return VLLMEngine(model, **kwargs)


