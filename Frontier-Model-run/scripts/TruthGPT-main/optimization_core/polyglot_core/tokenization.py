"""
Unified Tokenization module with automatic backend selection.

Supports HuggingFace tokenizers with Rust acceleration when available.
Provides high-performance tokenization with automatic fallback.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
import numpy as np

from .backend import Backend, get_best_backend, is_backend_available

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default tokenizer parameters
DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_MAX_LENGTH = 512

# Tensor types
TENSOR_TYPE_NUMPY = "np"
TENSOR_TYPE_PYTORCH = "pt"
TENSOR_TYPE_TENSORFLOW = "tf"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenizerConfig:
    """
    Configuration for tokenizer.
    
    Attributes:
        model_name: Model name or path
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        return_tensors: Tensor type to return (np, pt, tf)
        add_special_tokens: Whether to add special tokens
    """
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = DEFAULT_MAX_LENGTH
    padding: bool = True
    truncation: bool = True
    return_tensors: str = TENSOR_TYPE_NUMPY
    add_special_tokens: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        valid_tensor_types = [TENSOR_TYPE_NUMPY, TENSOR_TYPE_PYTORCH, TENSOR_TYPE_TENSORFLOW]
        if self.return_tensors not in valid_tensor_types:
            raise ValueError(
                f"return_tensors must be one of {valid_tensor_types}, "
                f"got {self.return_tensors}"
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Tokenizer:
    """
    Unified Tokenizer with automatic backend selection.
    
    Automatically selects the best backend:
    - Rust: Fast tokenization via HuggingFace tokenizers-rs (10-100x faster)
    - Python: HuggingFace transformers (fallback)
    
    Example:
        >>> tokenizer = Tokenizer(model_name="gpt2")
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(tokens)
        >>> print(f"Vocab size: {tokenizer.vocab_size}")
    """
    
    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        backend: Optional[Backend] = None,
        **kwargs
    ):
        """
        Initialize Tokenizer.
        
        Args:
            config: Tokenizer configuration
            model_name: Model name (if config not provided)
            backend: Force specific backend (None = auto-select)
            **kwargs: Additional config options
            
        Raises:
            RuntimeError: If tokenizer cannot be loaded
        """
        if config is None:
            config = TokenizerConfig(model_name=model_name, **kwargs)
        
        self.config = config
        self._backend = backend or get_best_backend('tokenization')
        self._impl = self._create_implementation()
    
    def _create_implementation(self):
        """
        Create backend-specific implementation.
        
        Returns:
            Tokenizer implementation
            
        Raises:
            RuntimeError: If no backend can be created
        """
        if self._backend == Backend.RUST and is_backend_available(Backend.RUST):
            rust_impl = self._create_rust_impl()
            if rust_impl is not None:
                return rust_impl
        
        # Fallback to Python
        return self._create_python_impl()
    
    def _create_rust_impl(self):
        """
        Create Rust implementation.
        
        Returns:
            Rust tokenizer or None if unavailable
        """
        try:
            from optimization_core.rust_core import truthgpt_rust
            return truthgpt_rust.PyTokenizer(self.config.model_name)
        except (ImportError, AttributeError, Exception) as e:
            print(f"[Tokenizer] Rust backend failed: {e}, using Python fallback")
            return None
    
    def _create_python_impl(self):
        """
        Create Python implementation.
        
        Returns:
            HuggingFace tokenizer
            
        Raises:
            RuntimeError: If tokenizer cannot be loaded
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Set pad_token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
        except ImportError:
            raise RuntimeError(
                "transformers library not installed. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer '{self.config.model_name}': {e}")
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: Optional[bool] = None,
        max_length: Optional[int] = None,
        padding: Optional[bool] = None,
        truncation: Optional[bool] = None
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Add special tokens (default: from config)
            max_length: Maximum length (default: from config)
            padding: Pad sequences (default: from config)
            truncation: Truncate sequences (default: from config)
            
        Returns:
            Token IDs (single list) or list of token ID lists (batch)
            
        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokens = tokenizer.encode("Hello, world!")
            >>> batch_tokens = tokenizer.encode(["Hello", "World"])
        """
        # Use config defaults if not specified
        add_special = (
            add_special_tokens 
            if add_special_tokens is not None 
            else self.config.add_special_tokens
        )
        max_len = max_length or self.config.max_length
        pad = padding if padding is not None else self.config.padding
        trunc = truncation if truncation is not None else self.config.truncation
        
        # Handle Rust backend
        if self._impl is not None and self._backend == Backend.RUST:
            return self._encode_rust(text, add_special, max_len, pad, trunc)
        
        # Handle Python backend
        return self._encode_python(text, add_special, max_len, pad, trunc)
    
    def _encode_rust(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool,
        max_length: int,
        padding: bool,
        truncation: bool
    ) -> Union[List[int], List[List[int]]]:
        """Encode using Rust backend."""
        if isinstance(text, str):
            return self._impl.encode(text, add_special_tokens, max_length, padding, truncation)
        else:
            return [
                self._impl.encode(t, add_special_tokens, max_length, padding, truncation)
                for t in text
            ]
    
    def _encode_python(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool,
        max_length: int,
        padding: bool,
        truncation: bool
    ) -> Union[List[int], List[List[int]]]:
        """Encode using Python backend."""
        if isinstance(text, str):
            result = self._impl(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None
            )
            return result['input_ids']
        else:
            result = self._impl(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None
            )
            return result['input_ids'].tolist()
    
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs or list of token ID lists
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text or list of texts
            
        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokens = [15496, 11, 995, 0]
            >>> text = tokenizer.decode(tokens)
            >>> print(text)  # "Hello, world!"
        """
        # Handle Rust backend
        if self._impl is not None and self._backend == Backend.RUST:
            return self._decode_rust(token_ids, skip_special_tokens)
        
        # Handle Python backend
        return self._decode_python(token_ids, skip_special_tokens)
    
    def _decode_rust(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool
    ) -> Union[str, List[str]]:
        """Decode using Rust backend."""
        if isinstance(token_ids[0], list):
            return [
                self._impl.decode(ids, skip_special_tokens)
                for ids in token_ids
            ]
        else:
            return self._impl.decode(token_ids, skip_special_tokens)
    
    def _decode_python(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool
    ) -> Union[str, List[str]]:
        """Decode using Python backend."""
        if isinstance(token_ids[0], list):
            return self._impl.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
        else:
            return self._impl.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens (not IDs).
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
            
        Examples:
            >>> tokenizer = Tokenizer()
            >>> tokens = tokenizer.tokenize("Hello, world!")
            >>> print(tokens)  # ['Hello', ',', ' world', '!']
        """
        if self._impl is not None:
            return self._impl.tokenize(text)
        else:
            raise RuntimeError("Tokenizer implementation not available")
    
    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of tokens in vocabulary
        """
        if self._impl is not None:
            if hasattr(self._impl, 'vocab_size'):
                return self._impl.vocab_size
            elif hasattr(self._impl, 'vocab'):
                return len(self._impl.vocab)
            elif hasattr(self._impl, 'get_vocab'):
                return len(self._impl.get_vocab())
        return 0
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """
        Get padding token ID.
        
        Returns:
            Padding token ID or None
        """
        if self._impl is not None:
            if hasattr(self._impl, 'pad_token_id'):
                return self._impl.pad_token_id
            elif hasattr(self._impl, 'pad_token') and self._impl.pad_token:
                return self._impl.convert_tokens_to_ids(self._impl.pad_token)
        return None
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """
        Get end-of-sequence token ID.
        
        Returns:
            EOS token ID or None
        """
        if self._impl is not None:
            if hasattr(self._impl, 'eos_token_id'):
                return self._impl.eos_token_id
            elif hasattr(self._impl, 'eos_token') and self._impl.eos_token:
                return self._impl.convert_tokens_to_ids(self._impl.eos_token)
        return None
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """
        Get beginning-of-sequence token ID.
        
        Returns:
            BOS token ID or None
        """
        if self._impl is not None:
            if hasattr(self._impl, 'bos_token_id'):
                return self._impl.bos_token_id
            elif hasattr(self._impl, 'bos_token') and self._impl.bos_token:
                return self._impl.convert_tokens_to_ids(self._impl.bos_token)
        return None
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """
        Get unknown token ID.
        
        Returns:
            UNK token ID or None
        """
        if self._impl is not None:
            if hasattr(self._impl, 'unk_token_id'):
                return self._impl.unk_token_id
            elif hasattr(self._impl, 'unk_token') and self._impl.unk_token:
                return self._impl.convert_tokens_to_ids(self._impl.unk_token)
        return None
    
    @property
    def backend(self) -> Backend:
        """Get current backend."""
        return self._backend
    
    def __repr__(self) -> str:
        return (f"Tokenizer(model={self.config.model_name}, "
                f"vocab_size={self.vocab_size}, "
                f"backend={self._backend.name})")

