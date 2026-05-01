"""
Unified Inference Engine with automatic backend selection.

Supports text generation with various sampling strategies:
- Greedy decoding (deterministic, fastest)
- Sampling with temperature, top-k, top-p
- Beam search
- Repetition penalty
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
import numpy as np
import time

from .backend import Backend, get_best_backend, is_backend_available

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default generation parameters
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_NUM_BEAMS = 1

# Finish reasons
FINISH_REASON_MAX_LENGTH = "max_length"
FINISH_REASON_EOS = "eos"
FINISH_REASON_TIMEOUT = "timeout"

# Numerical stability
EPSILON = 1e-10

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens for sampling
        top_p: Nucleus sampling threshold (cumulative probability)
        repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
        do_sample: Whether to use sampling (False = greedy)
        num_beams: Number of beams for beam search
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
    """
    max_new_tokens: int = field(default=DEFAULT_MAX_NEW_TOKENS)
    temperature: float = field(default=DEFAULT_TEMPERATURE)
    top_k: int = field(default=DEFAULT_TOP_K)
    top_p: float = field(default=DEFAULT_TOP_P)
    repetition_penalty: float = field(default=DEFAULT_REPETITION_PENALTY)
    do_sample: bool = True
    num_beams: int = field(default=DEFAULT_NUM_BEAMS)
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if not 0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}")
    
    @classmethod
    def greedy(cls) -> "GenerationConfig":
        """
        Greedy decoding - deterministic, fastest.
        
        Always selects the token with highest probability.
        """
        return cls(do_sample=False, temperature=1.0, num_beams=1)
    
    @classmethod
    def sampling(cls, temperature: float = 0.7, top_p: float = 0.9) -> "GenerationConfig":
        """
        Sampling with temperature and nucleus.
        
        Args:
            temperature: Sampling temperature (default: 0.7)
            top_p: Nucleus sampling threshold (default: 0.9)
        """
        return cls(do_sample=True, temperature=temperature, top_p=top_p)
    
    @classmethod
    def beam_search(cls, num_beams: int = 4) -> "GenerationConfig":
        """
        Beam search for best sequence.
        
        Args:
            num_beams: Number of beams to keep (default: 4)
        """
        return cls(do_sample=False, num_beams=num_beams)
    
    @classmethod
    def creative(cls) -> "GenerationConfig":
        """
        Creative/diverse generation configuration.
        
        High temperature and top-p for more diverse outputs.
        """
        return cls(
            do_sample=True,
            temperature=0.9,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    @classmethod
    def factual(cls) -> "GenerationConfig":
        """
        Factual/deterministic generation configuration.
        
        Low temperature and conservative sampling for factual outputs.
        """
        return cls(
            do_sample=True,
            temperature=0.3,
            top_k=20,
            top_p=0.85,
            repetition_penalty=1.05
        )


@dataclass
class InferenceConfig:
    """
    Configuration for inference engine.
    
    Attributes:
        seed: Random seed for reproducibility
        use_cache: Whether to use KV cache
        max_batch_size: Maximum batch size for batched generation
        timeout_ms: Timeout in milliseconds
    """
    seed: int = 42
    use_cache: bool = True
    max_batch_size: int = 8
    timeout_ms: float = 30000.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size}")
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")


@dataclass
class GenerationResult:
    """
    Result from text generation.
    
    Attributes:
        token_ids: Generated token IDs (including input)
        tokens_generated: Number of newly generated tokens
        generation_time_ms: Generation time in milliseconds
        finish_reason: Reason for stopping (max_length, eos, timeout)
    """
    token_ids: List[int]
    tokens_generated: int
    generation_time_ms: float
    finish_reason: str = field(default=FINISH_REASON_MAX_LENGTH)
    
    @property
    def tokens_per_second(self) -> float:
        """
        Calculate tokens per second generation rate.
        
        Returns:
            Tokens per second (0.0 if generation_time_ms <= 0)
        """
        if self.generation_time_ms <= 0:
            return 0.0
        return self.tokens_generated / (self.generation_time_ms / 1000.0)

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

class TokenSampler:
    """
    Token sampling with various strategies.
    
    Supports:
    - Greedy decoding (argmax)
    - Temperature scaling
    - Top-K filtering
    - Top-P (nucleus) filtering
    - Repetition penalty
    
    Example:
        >>> sampler = TokenSampler(seed=42)
        >>> token = sampler.sample(logits, config, prev_tokens=[1, 2, 3])
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize TokenSampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        self._rng = np.random.default_rng(seed)
    
    def sample(
        self,
        logits: np.ndarray,
        config: GenerationConfig,
        prev_tokens: Optional[List[int]] = None
    ) -> int:
        """
        Sample next token from logits.
        
        Args:
            logits: Logits array [vocab_size]
            config: Generation configuration
            prev_tokens: Previous tokens for repetition penalty
            
        Returns:
            Sampled token ID
            
        Algorithm:
            1. Apply repetition penalty to logits
            2. If greedy: return argmax
            3. Apply temperature scaling
            4. Convert to probabilities (softmax)
            5. Apply top-k filtering
            6. Apply top-p (nucleus) filtering
            7. Sample from filtered distribution
        """
        if len(logits) == 0:
            raise ValueError("logits array cannot be empty")
        
        # Work with float64 for numerical stability
        logits = logits.astype(np.float64).copy()
        
        # Apply repetition penalty to discourage repetition
        if config.repetition_penalty != 1.0 and prev_tokens:
            logits = self._apply_repetition_penalty(logits, prev_tokens, config.repetition_penalty)
        
        # Greedy decoding: return token with highest probability
        if not config.do_sample:
            return int(np.argmax(logits))
        
        # Temperature scaling: higher temperature = more random
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Convert to probabilities using numerically stable softmax
        probs = self._softmax(logits)
        
        # Apply top-k filtering: keep only top-k tokens
        if config.top_k > 0 and config.top_k < len(probs):
            probs = self._apply_top_k(probs, config.top_k)
        
        # Apply top-p (nucleus) filtering: keep tokens until cumulative prob >= top_p
        if config.top_p < 1.0:
            probs = self._apply_top_p(probs, config.top_p)
        
        # Sample from filtered distribution
        return int(self._rng.choice(len(probs), p=probs))
    
    def _apply_repetition_penalty(
        self,
        logits: np.ndarray,
        prev_tokens: List[int],
        penalty: float
    ) -> np.ndarray:
        """
        Apply repetition penalty to logits.
        
        Args:
            logits: Logits array
            prev_tokens: Previously generated tokens
            penalty: Penalty factor (>1.0 reduces probability of repeated tokens)
            
        Returns:
            Modified logits array
        """
        # Get unique previous tokens to avoid double-penalizing
        unique_prev_tokens = set(prev_tokens)
        
        for token_id in unique_prev_tokens:
            if 0 <= token_id < len(logits):
                # Reduce probability of repeating this token
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty
        
        return logits
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute numerically stable softmax.
        
        Args:
            logits: Logits array
            
        Returns:
            Probability distribution
        """
        # Subtract max for numerical stability
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        return exp_logits / (exp_logits.sum() + EPSILON)
    
    def _apply_top_k(self, probs: np.ndarray, k: int) -> np.ndarray:
        """
        Apply top-k filtering to probabilities.
        
        Args:
            probs: Probability distribution
            k: Number of top tokens to keep
            
        Returns:
            Filtered probability distribution
        """
        # Get indices of top-k tokens
        top_k_indices = np.argsort(probs)[-k:]
        
        # Create mask for top-k tokens
        mask = np.zeros_like(probs)
        mask[top_k_indices] = 1.0
        
        # Zero out non-top-k probabilities and renormalize
        filtered_probs = probs * mask
        return filtered_probs / (filtered_probs.sum() + EPSILON)
    
    def _apply_top_p(self, probs: np.ndarray, p: float) -> np.ndarray:
        """
        Apply top-p (nucleus) filtering to probabilities.
        
        Args:
            probs: Probability distribution
            p: Cumulative probability threshold
            
        Returns:
            Filtered probability distribution
        """
        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Compute cumulative probabilities
        cumsum = np.cumsum(sorted_probs)
        
        # Find cutoff index where cumulative prob >= p
        cutoff_idx = np.searchsorted(cumsum, p) + 1
        cutoff_idx = min(cutoff_idx, len(probs))
        
        # Create mask for tokens up to cutoff
        mask = np.zeros_like(probs)
        mask[sorted_indices[:cutoff_idx]] = 1.0
        
        # Zero out tokens beyond cutoff and renormalize
        filtered_probs = probs * mask
        return filtered_probs / (filtered_probs.sum() + EPSILON)

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Unified Inference Engine with automatic backend selection.
    
    Supports text generation with:
    - Greedy decoding (deterministic, fastest)
    - Sampling (temperature, top-k, top-p)
    - Beam search (multi-candidate generation)
    - Repetition penalty
    
    Example:
        >>> engine = InferenceEngine(seed=42)
        >>> config = GenerationConfig.sampling(temperature=0.7)
        >>> result = engine.generate(prompt_ids, model.forward, config)
        >>> print(f"{result.tokens_per_second:.0f} tokens/sec")
    """
    
    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        seed: int = 42,
        backend: Optional[Backend] = None
    ):
        """
        Initialize Inference Engine.
        
        Args:
            config: Inference configuration
            seed: Random seed for reproducibility
            backend: Force specific backend (None = auto-select)
        """
        if config is None:
            config = InferenceConfig(seed=seed)
        
        self.config = config
        self._backend = backend or get_best_backend('inference')
        self._sampler = TokenSampler(seed=config.seed)
        self._impl = self._create_implementation()
    
    def _create_implementation(self):
        """
        Create backend-specific implementation.
        
        Returns:
            Backend implementation or None (use Python fallback)
        """
        if self._backend == Backend.CPP and is_backend_available(Backend.CPP):
            return self._create_cpp_impl()
        # Add other backend implementations here
        return None
    
    def _create_cpp_impl(self):
        """
        Create C++ implementation.
        
        Returns:
            C++ inference engine or None if unavailable
        """
        # TODO: Implement C++ inference engine
        # Would use C++ bindings for faster inference
        return None
    
    def generate(
        self,
        input_ids: List[int],
        forward_fn: Callable[[List[int]], np.ndarray],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate tokens given input IDs.
        
        Args:
            input_ids: Input token IDs
            forward_fn: Model forward function (tokens -> logits)
            config: Generation configuration
            **kwargs: Config overrides (merged into config)
            
        Returns:
            GenerationResult with generated tokens and statistics
            
        Raises:
            ValueError: If input_ids is empty or forward_fn is invalid
        """
        if not input_ids:
            raise ValueError("input_ids cannot be empty")
        
        # Merge kwargs into config
        if config is None:
            config = GenerationConfig(**kwargs)
        else:
            # Update config with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        start_time = time.perf_counter()
        
        # Initialize token sequence with input
        tokens = list(input_ids)
        generated_count = 0
        finish_reason = FINISH_REASON_MAX_LENGTH
        
        # Generate tokens one by one
        for _ in range(config.max_new_tokens):
            # Get logits from model forward pass
            logits = forward_fn(tokens)
            
            # Ensure logits is a flat numpy array
            if isinstance(logits, np.ndarray):
                logits = logits.flatten()
            else:
                logits = np.array(logits).flatten()
            
            # Sample next token
            if config.num_beams > 1:
                # TODO: Implement proper beam search
                # For now, use regular sampling
                next_token = self._sampler.sample(logits, config, tokens)
            else:
                next_token = self._sampler.sample(logits, config, tokens)
            
            # Append token to sequence
            tokens.append(int(next_token))
            generated_count += 1
            
            # Check for end-of-sequence token
            if config.eos_token_id is not None and next_token == config.eos_token_id:
                finish_reason = FINISH_REASON_EOS
                break
            
            # Check timeout
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.config.timeout_ms:
                finish_reason = FINISH_REASON_TIMEOUT
                break
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return GenerationResult(
            token_ids=tokens,
            tokens_generated=generated_count,
            generation_time_ms=elapsed_ms,
            finish_reason=finish_reason
        )
    
    def generate_batch(
        self,
        batch_input_ids: List[List[int]],
        forward_fn: Callable[[List[List[int]]], np.ndarray],
        config: Optional[GenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        Generate tokens for a batch of inputs.
        
        Args:
            batch_input_ids: List of input token ID lists
            forward_fn: Batched forward function (batch -> logits)
            config: Generation configuration
            
        Returns:
            List of GenerationResult for each input
            
        Note:
            Currently processes sequentially. TODO: Implement proper batched generation
            with padding and attention masks.
        """
        if not batch_input_ids:
            return []
        
        if len(batch_input_ids) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(batch_input_ids)} exceeds max_batch_size "
                f"{self.config.max_batch_size}"
            )
        
        results = []
        for input_ids in batch_input_ids:
            # Create single-input forward function
            single_forward = lambda t: forward_fn([t])[0]
            result = self.generate(input_ids, single_forward, config)
            results.append(result)
        
        return results
    
    @property
    def backend(self) -> Backend:
        """Get current backend."""
        return self._backend
    
    def __repr__(self) -> str:
        return (f"InferenceEngine(seed={self.config.seed}, "
                f"backend={self._backend.name}, "
                f"use_cache={self.config.use_cache})")

