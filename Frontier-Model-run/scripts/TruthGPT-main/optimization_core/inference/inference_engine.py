"""
Professional inference engine with batching, caching, and optimization.
"""
import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.cuda.amp import autocast

from .base_engine import BaseInferenceEngine
from .utils.validators import (
    validate_generation_params,
    validate_positive_int,
)
from .utils.prompt_utils import (
    normalize_prompts,
    handle_single_prompt,
)
from .utils.decorators import (
    log_execution_time,
    handle_errors,
)
from .utils.logging_utils import (
    log_operation,
    MetricsCollector,
)

logger = logging.getLogger(__name__)


class InferenceEngine(BaseInferenceEngine):
    """
    Professional inference engine with optimizations.
    Supports batching, caching, and mixed precision.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        max_batch_size: int = 8,
        max_seq_length: int = 512,
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer
            device: Device for inference
            use_amp: Use automatic mixed precision
            amp_dtype: AMP dtype
            max_batch_size: Maximum batch size
            max_seq_length: Maximum sequence length
        """
        # Initialize base class with model name/path if available
        model_name = getattr(model, 'name_or_path', 'unknown')
        super().__init__(model=model_name)
        
        # Validate parameters using shared validators
        validate_positive_int(max_batch_size, "max_batch_size")
        validate_positive_int(max_seq_length, "max_seq_length")
        
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = amp_dtype
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        
        # Move model to device if needed
        if device != next(model.parameters()).device:
            self.model = self.model.to(device)
        
        self.model.eval()
        self._set_initialized(True)
        logger.info(f"Inference engine initialized on {device}")
    
    def _initialize_engine(self, **kwargs) -> Any:
        """Initialize the underlying engine (not used for PyTorch models)."""
        return self.model
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_beams: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts with batching support.
        
        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Use sampling
            num_beams: Number of beams for beam search
            **kwargs: Additional generation arguments
        
        Returns:
            Generated text(s)
        """
        # Validate generation parameters using shared validators
        validate_generation_params(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Normalize prompts using shared utilities
        prompts_list, was_single = normalize_prompts(prompts)
        
        try:
            # Tokenize with batching
            inputs = self.tokenizer(
                prompts_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.device)
            
            # Generate with AMP
            with torch.no_grad():
                if self.use_amp:
                    with autocast(dtype=self.amp_dtype):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p if do_sample else None,
                            top_k=top_k if do_sample else None,
                            repetition_penalty=repetition_penalty,
                            num_beams=num_beams if not do_sample else 1,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            **kwargs
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p if do_sample else None,
                        top_k=top_k if do_sample else None,
                        repetition_penalty=repetition_penalty,
                        num_beams=num_beams if not do_sample else 1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
            
            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            # Remove input prompts
            results = []
            for i, (prompt, generated) in enumerate(zip(prompts_list, generated_texts)):
                if generated.startswith(prompt):
                    result = generated[len(prompt):].strip()
                else:
                    result = generated.strip()
                results.append(result if result else prompt)
            
            return handle_single_prompt(results, was_single)
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory during generation")
            raise RuntimeError(
                "GPU out of memory. Try reducing max_new_tokens or batch size."
            )
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            raise
    
    def generate_batched(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of prompts
            batch_size: Batch size (defaults to max_batch_size)
            **generation_kwargs: Generation arguments
        
        Returns:
            List of generated texts
        """
        batch_size = batch_size or self.max_batch_size
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.generate(batch, **generation_kwargs)
            results.extend(batch_results)
        
        return results
    
    def profile_inference(
        self,
        prompt: str,
        num_runs: int = 10,
        warmup_runs: int = 3,
        **generation_kwargs
    ) -> Dict[str, float]:
        """
        Profile inference performance.
        
        Args:
            prompt: Test prompt
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            **generation_kwargs: Generation arguments
        
        Returns:
            Dictionary with performance metrics
        """
        # Warmup
        for _ in range(warmup_runs):
            _ = self.generate(prompt, **generation_kwargs)
        
        # Synchronize
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Profile
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.generate(prompt, **generation_kwargs)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": 1.0 / avg_time,  # samples per second
        }


class BatchProcessor:
    """
    Handles efficient batch processing with dynamic batching.
    """
    
    def __init__(
        self,
        inference_engine: InferenceEngine,
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,
    ):
        """
        Initialize batch processor.
        
        Args:
            inference_engine: Inference engine instance
            max_batch_size: Maximum batch size
            max_wait_time: Maximum wait time before processing batch
        """
        self.engine = inference_engine
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self._queue: List[Dict[str, Any]] = []
    
    def process(self, prompt: str, callback: Optional[callable] = None, **kwargs) -> str:
        """
        Process a single prompt (can be batched).
        
        Args:
            prompt: Input prompt
            callback: Optional callback function
            **kwargs: Generation arguments
        
        Returns:
            Generated text
        """
        # For now, process immediately
        # In production, this could implement proper batching with queue
        result = self.engine.generate(prompt, **kwargs)
        
        if callback:
            callback(result)
        
        return result
    
    def process_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Process a batch of prompts.
        
        Args:
            prompts: List of prompts
            **kwargs: Generation arguments
        
        Returns:
            List of generated texts
        """
        return self.engine.generate_batched(prompts, **kwargs)



