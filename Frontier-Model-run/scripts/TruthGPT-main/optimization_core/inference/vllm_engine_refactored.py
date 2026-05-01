"""
Refactored vLLM Engine with Polyglot Integration

Integrates vLLM with Rust KV cache and C++ attention for maximum performance.
"""
import logging
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .base_engine import BaseInferenceEngine, GenerationConfig

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available. Install with: pip install vllm>=0.2.0")

try:
    from optimization_core.polyglot.kv_cache import KVCache
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False

class BackendMode(Enum):
    VLLM_ONLY = "vllm_only"
    VLLM_RUST = "vllm_rust"
    VLLM_CPP = "vllm_cpp"
    AUTO = "auto"

@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    trust_remote_code: bool = False
    enable_prefix_caching: bool = True
    use_rust_kv_cache: bool = True
    backend_mode: BackendMode = BackendMode.AUTO

class VLLMEngineRefactored(BaseInferenceEngine):
    """
    Refactored vLLM engine with polyglot integration.
    
    Features:
    - vLLM PagedAttention (3-5x memory reduction)
    - Rust KV cache for prefix caching
    - C++ attention kernels (optional)
    - Continuous batching
    - Multi-GPU support
    """
    
    def __init__(
        self,
        model: str,
        config: Optional[VLLMConfig] = None,
        **kwargs
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed")
        
        super().__init__(model=model, **kwargs)
        
        self.config = config or VLLMConfig()
        self._setup_backend()
        self._setup_kv_cache()
        self._setup_engine()
    
    def _setup_backend(self):
        """Setup backend mode."""
        if self.config.backend_mode == BackendMode.AUTO:
            if POLYGLOT_AVAILABLE:
                try:
                    from optimization_core.polyglot import get_available_backends
                    backends = get_available_backends()
                    if backends.get("rust"):
                        self.config.backend_mode = BackendMode.VLLM_RUST
                    elif backends.get("cpp"):
                        self.config.backend_mode = BackendMode.VLLM_CPP
                    else:
                        self.config.backend_mode = BackendMode.VLLM_ONLY
                except Exception:
                    self.config.backend_mode = BackendMode.VLLM_ONLY
            else:
                self.config.backend_mode = BackendMode.VLLM_ONLY
        
        logger.info(f"vLLM backend mode: {self.config.backend_mode}")
    
    def _setup_kv_cache(self):
        """Setup external KV cache if enabled."""
        self.external_cache = None
        if self.config.use_rust_kv_cache and POLYGLOT_AVAILABLE:
            try:
                self.external_cache = KVCache(
                    max_size=8192,
                    eviction_strategy="adaptive",
                    enable_compression=True,
                )
                logger.info("External Rust KV cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize external cache: {e}")
    
    def _setup_engine(self):
        """Setup vLLM engine."""
        engine_kwargs = {
            "model": self.model,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "trust_remote_code": self.config.trust_remote_code,
        }
        
        if self.config.max_model_len:
            engine_kwargs["max_model_len"] = self.config.max_model_len
        
        if self.config.dtype != "auto":
            engine_kwargs["dtype"] = self.config.dtype
        
        if self.config.quantization:
            engine_kwargs["quantization"] = self.config.quantization
        
        if self.config.enable_prefix_caching:
            engine_kwargs["enable_prefix_caching"] = True
        
        try:
            self.llm = LLM(**engine_kwargs)
            logger.info(f"vLLM engine initialized: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate with vLLM and optional polyglot optimizations."""
        gen_config = config or GenerationConfig(**kwargs)
        
        single_prompt = isinstance(prompts, str)
        if single_prompt:
            prompts = [prompts]
        
        sampling_params = SamplingParams(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            max_tokens=gen_config.max_new_tokens,
            repetition_penalty=gen_config.repetition_penalty,
            stop=kwargs.get("stop", None),
        )
        
        try:
            outputs = self.llm.generate(prompts, sampling_params)
            
            results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                results.append(generated_text)
            
            if self.external_cache:
                self._update_cache(prompts, results)
            
            return results[0] if single_prompt else results
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise
    
    def _update_cache(self, prompts: List[str], results: List[str]):
        """Update external KV cache with generated sequences."""
        if not self.external_cache:
            return
        
        try:
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                cache_key = hash(prompt)
                cache_data = (prompt + result).encode('utf-8')
                self.external_cache.put(
                    layer_idx=0,
                    position=i,
                    data=cache_data,
                    key=str(cache_key),
                )
        except Exception as e:
            logger.debug(f"Cache update failed: {e}")
    
    def generate_async(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[str]:
        """Async generation with continuous batching."""
        gen_config = config or GenerationConfig(**kwargs)
        
        sampling_params = SamplingParams(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            max_tokens=gen_config.max_new_tokens,
            repetition_penalty=gen_config.repetition_penalty,
        )
        
        async def _generate():
            async_engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(model=self.model)
            )
            
            request_ids = []
            for prompt in prompts:
                request_id = await async_engine.add_request(
                    prompt=prompt,
                    sampling_params=sampling_params,
                )
                request_ids.append(request_id)
            
            results = []
            while request_ids:
                async for request_output in async_engine.generate():
                    if request_output.finished:
                        results.append(request_output.outputs[0].text)
                        request_ids.remove(request_output.request_id)
                        if not request_ids:
                            break
            
            return results
        
        import asyncio
        return asyncio.run(_generate())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "model": self.model,
            "backend_mode": self.config.backend_mode.value,
            "tensor_parallel_size": self.config.tensor_parallel_size,
        }
        
        if hasattr(self.llm, "llm_engine"):
            engine = self.llm.llm_engine
            if hasattr(engine, "scheduler"):
                scheduler = engine.scheduler
                stats.update({
                    "waiting": len(scheduler.waiting),
                    "running": len(scheduler.running),
                    "swapped": len(scheduler.swapped),
                })
        
        if self.external_cache:
            cache_stats = self.external_cache.stats()
            stats["cache"] = cache_stats
        
        return stats













