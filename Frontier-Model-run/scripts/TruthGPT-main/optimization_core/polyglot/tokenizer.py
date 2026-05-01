"""
Unified Tokenizer Interface

Provides Python interface to Rust tokenizer backend.
"""
from typing import Union, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    from truthgpt_rust import PyFastTokenizer
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

class Tokenizer:
    """
    Unified tokenizer interface.
    
    Automatically uses Rust backend when available (3x faster).
    Falls back to HuggingFace tokenizers otherwise.
    """
    
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        model_name: Optional[str] = None,
        use_rust: bool = True,
    ):
        self.tokenizer_path = tokenizer_path
        self.model_name = model_name
        self.use_rust = use_rust and RUST_AVAILABLE
        
        self.rust_tokenizer = None
        self.python_tokenizer = None
        
        self._setup_backends()
    
    def _setup_backends(self):
        """Setup tokenizer backends."""
        if self.use_rust:
            try:
                if self.tokenizer_path:
                    self.rust_tokenizer = PyFastTokenizer(self.tokenizer_path)
                elif self.model_name:
                    self.rust_tokenizer = PyFastTokenizer.from_pretrained(self.model_name)
                
                if self.rust_tokenizer:
                    logger.info("Rust tokenizer initialized")
                    return
            except Exception as e:
                logger.warning(f"Failed to initialize Rust tokenizer: {e}")
        
        try:
            from transformers import AutoTokenizer
            if self.model_name:
                self.python_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            elif self.tokenizer_path:
                self.python_tokenizer = AutoTokenizer.from_file(self.tokenizer_path)
            
            logger.info("Python tokenizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Python tokenizer: {e}")
            raise
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, Any]]:
        """Encode text to token IDs."""
        single_text = isinstance(text, str)
        if single_text:
            text = [text]
        
        if self.rust_tokenizer:
            try:
                token_ids = self.rust_tokenizer.encode_batch(text, add_special_tokens)
                
                if return_tensors == "pt":
                    import torch
                    max_len = max(len(ids) for ids in token_ids)
                    padded = []
                    attention_mask = []
                    
                    for ids in token_ids:
                        pad_len = max_len - len(ids)
                        pad_id = self.python_tokenizer.pad_token_id if self.python_tokenizer else 0
                        padded.append(ids + [pad_id] * pad_len)
                        attention_mask.append([1] * len(ids) + [0] * pad_len)
                    
                    return {
                        "input_ids": torch.tensor(padded),
                        "attention_mask": torch.tensor(attention_mask),
                    }
                elif return_tensors is None:
                    return token_ids[0] if single_text else token_ids
                else:
                    return token_ids
            except Exception as e:
                logger.warning(f"Rust encoding failed: {e}, falling back")
        
        if self.python_tokenizer:
            result = self.python_tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=True,
                truncation=True,
            )
            return result if not single_text or return_tensors else result["input_ids"][0]
        
        raise RuntimeError("No tokenizer available")
    
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """Decode token IDs to text."""
        single_sequence = isinstance(token_ids[0], int) if token_ids else True
        
        if single_sequence and isinstance(token_ids[0], int):
            token_ids = [token_ids]
        
        if self.rust_tokenizer:
            try:
                texts = self.rust_tokenizer.decode_batch(
                    [list(ids) for ids in token_ids],
                    skip_special_tokens
                )
                return texts[0] if single_sequence else texts
            except Exception as e:
                logger.warning(f"Rust decoding failed: {e}, falling back")
        
        if self.python_tokenizer:
            if single_sequence:
                return self.python_tokenizer.decode(
                    token_ids[0],
                    skip_special_tokens=skip_special_tokens
                )
            else:
                return self.python_tokenizer.batch_decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens
                )
        
        raise RuntimeError("No tokenizer available")
    
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.rust_tokenizer:
            return self.rust_tokenizer.vocab_size()
        elif self.python_tokenizer:
            return len(self.python_tokenizer)
        return 0
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        if self.rust_tokenizer:
            return self.rust_tokenizer.token_to_id(token)
        elif self.python_tokenizer:
            return self.python_tokenizer.convert_tokens_to_ids(token)
        return None
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert ID to token."""
        if self.rust_tokenizer:
            return self.rust_tokenizer.id_to_token(token_id)
        elif self.python_tokenizer:
            return self.python_tokenizer.convert_ids_to_tokens(token_id)
        return None

def create_tokenizer(
    tokenizer_path: Optional[str] = None,
    model_name: Optional[str] = None,
    use_rust: bool = True,
) -> Tokenizer:
    """Factory function to create tokenizer."""
    return Tokenizer(
        tokenizer_path=tokenizer_path,
        model_name=model_name,
        use_rust=use_rust,
    )

__all__ = ["Tokenizer", "create_tokenizer"]













