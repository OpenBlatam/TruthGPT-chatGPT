"""
Inference service for model inference operations.
"""
import logging
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict
import torch

from .base_service import BaseService
from ..event_system import EventType
from ...inference.text_generator import TextGenerator
from ...inference.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class InferenceConfig(BaseModel):
    """Configuration for inference environment parameters."""
    device: Optional[str] = Field(default=None, description="Target compute device (e.g., 'cuda:0', 'cpu')")
    use_cache: bool = Field(default=True, description="Enable KV caching")
    cache_dir: Optional[str] = Field(default=None, description="Cache object persistence directory")
    max_batch_size: int = Field(default=8, ge=1, le=1024, description="Maximum batches allowed during computation")
    max_seq_length: int = Field(default=512, ge=1, description="Context window total limit per inference request")

class GenerationConfig(BaseModel):
    """Configuration for text generation sampling parameters."""
    max_new_tokens: int = Field(default=64, ge=1, description="Limit for generated tokens output")
    temperature: float = Field(default=0.8, ge=0.0, description="Randomness of token generation sampling")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling probability mass constraint")
    top_k: int = Field(default=50, ge=0, description="Top-k tokens filtering limit")
    repetition_penalty: float = Field(default=1.1, ge=1.0, description="Multiplier penalty applied against repeating recent tokens")
    do_sample: bool = Field(default=True, description="Flag determining deterministic (greedy) or probabilistic generation")

class InferenceService(BaseService):
    """
    Service for inference operations.
    Handles text generation, batch processing, and caching.
    Supports strict validation configuration models.
    """
    
    def __init__(self, **kwargs):
        """Initialize inference service."""
        super().__init__(name="InferenceService", **kwargs)
        self.text_generator: Optional[TextGenerator] = None
        self.inference_engine: Optional[InferenceEngine] = None
    
    def _do_initialize(self) -> None:
        """Initialize inference components."""
        pass
    
    def configure(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: Optional[Union[Dict[str, Any], InferenceConfig]] = None,
    ) -> None:
        """
        Configure inference service.
        
        Args:
            model: Model for inference
            tokenizer: Tokenizer
            config: Optional inference configuration
        """
        # Adopt Pydantic Model
        if config is None:
            config = InferenceConfig()
        elif isinstance(config, dict):
            config = InferenceConfig(**config)
        
        # Create text generator with caching
        self.text_generator = TextGenerator(
            model=model,
            tokenizer=tokenizer,
            device=config.device,
            use_cache=config.use_cache,
            cache_dir=config.cache_dir,
            max_batch_size=config.max_batch_size,
            max_seq_length=config.max_seq_length,
        )
        
        # Store inference engine reference
        self.inference_engine = self.text_generator.engine
        
        logger.info("Inference service configured")
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[Union[Dict[str, Any], GenerationConfig]] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).
        
        Args:
            prompt: Input prompt(s)
            config: Optional generation configuration
            use_cache: Override cache setting
        
        Returns:
            Generated text(s)
        """
        if not self.text_generator:
            raise RuntimeError("Inference service not configured")
        
        if config is None:
            config = GenerationConfig()
        elif isinstance(config, dict):
            config = GenerationConfig(**config)
            
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": config.do_sample,
        }
        
        try:
            # Handle single vs batch
            if isinstance(prompt, str):
                result = self.text_generator.generate(
                    prompt,
                    use_cache=use_cache,
                    **generation_kwargs
                )
            else:
                result = self.text_generator.generate_batch(
                    prompt,
                    **generation_kwargs
                )
            
            # Emit event
            self.emit(EventType.METRIC_LOGGED, {
                "operation": "generate",
                "prompt_count": 1 if isinstance(prompt, str) else len(prompt),
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            self.emit(EventType.ERROR_OCCURRED, {
                "error": str(e),
                "operation": "generate",
            })
            raise
    
    def profile(
        self,
        prompt: str,
        config: Optional[Union[Dict[str, Any], GenerationConfig]] = None,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Profile inference performance.
        
        Args:
            prompt: Test prompt
            config: Optional generation configuration
            num_runs: Number of profiling runs
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_engine:
            raise RuntimeError("Inference service not configured")
        
        if config is None:
            config = GenerationConfig()
        elif isinstance(config, dict):
            config = GenerationConfig(**config)
            
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
        }
        
        metrics = self.inference_engine.profile_inference(
            prompt=prompt,
            num_runs=num_runs,
            **generation_kwargs
        )
        
        return metrics
    
    def clear_cache(self) -> None:
        """Clear inference cache."""
        if self.text_generator:
            self.text_generator.clear_cache()
            logger.info("Inference cache cleared")
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.text_generator:
            return self.text_generator.get_cache_stats()
        return None



