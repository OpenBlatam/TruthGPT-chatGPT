"""
Transformer-specific optimization techniques for LLMs
Following best practices for transformer optimization
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

# Import from the core package of optimization_core
from optimization_core.modules.optimizers.core.pytorch_optimizer_base import OptimizationConfig, PyTorchOptimizerBase

logger = logging.getLogger(__name__)


class TransformerOptimizer(PyTorchOptimizerBase):
    """
    Advanced transformer optimizer with LLM-specific techniques.
    
    Attributes:
        config (OptimizationConfig): Configuration object.
        model_name (str): Name or path of the model.
        tokenizer (PreTrainedTokenizer): Tokenizer instance.
        model (PreTrainedModel): Model instance.
    """

    def __init__(self, config: OptimizationConfig, model_name: Optional[str] = None):
        """
        Initialize the Transformer Optimizer.

        Args:
            config: Optimization configuration.
            model_name: Optional model name override. If None, uses config.model_name.
        """
        super().__init__(config)
        self.model_name = model_name or config.model_name
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.attention_cache: Dict[str, Any] = {}
        self.use_flash_attention: bool = False
        self.gradient_accumulation_steps: int = config.gradient_accumulation_steps

        # Initialize model and tokenizer
        self._initialize_model()

        # Transformer-specific optimizations
        self._setup_transformer_optimizations()

    def _initialize_model(self) -> None:
        """Initialize transformer model and tokenizer."""
        try:
            self.logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info(f"Loading model: {self.model_name}")
            # Use AutoModelForCausalLM for generation capabilities
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32,
                    trust_remote_code=True
                )
            except Exception:
                self.logger.warning(f"Could not load as CausalLM, falling back to generic AutoModel for {self.model_name}")
                # Fallback implementation
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.config.use_mixed_precision else torch.float32,
                    trust_remote_code=True
                )

            # Move to device
            self.model.to(self.device)

            self.logger.info(f"Initialized model and moved to {self.device}")

        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def _setup_transformer_optimizations(self) -> None:
        """Setup transformer-specific optimizations."""
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # Setup attention optimizations
        self._setup_attention_optimizations()

        # Setup memory optimizations
        self._setup_memory_optimizations()

    def _setup_attention_optimizations(self) -> None:
        """Setup attention mechanism optimizations."""
        # Flash Attention detection
        # PyTorch 2.0+ supports scaled_dot_product_attention natively
        if hasattr(F, "scaled_dot_product_attention"):
            self.use_flash_attention = True
            self.logger.info("Flash Attention (scaled_dot_product_attention) is available.")
        else:
            self.use_flash_attention = False
            self.logger.info("Flash Attention not available.")

        # Attention caching for inference
        self.attention_cache = {}

    def _setup_memory_optimizations(self) -> None:
        """Setup memory optimization techniques."""
        # Enable memory efficient attention (PyTorch backend)
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
        except Exception as e:
            self.logger.debug(f"Could not enable flash sdp backend: {e}")

        # Ensure gradient accumulation matches config
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with transformer optimizations.

        Args:
            input_ids: Tokenized input sequences.
            attention_mask: Attention mask for padding.
            labels: Target labels for training.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with model outputs.
        """
        outputs = {}

        try:
            # Prepare inputs
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            
            # Combine kwargs into inputs (e.g. past_key_values)
            model_inputs.update(kwargs)
            
            if labels is not None:
                model_inputs["labels"] = labels

            # Forward pass with optimizations
            if self.config.use_mixed_precision:
                with autocast():
                    model_outputs = self.model(**model_inputs)
            else:
                model_outputs = self.model(**model_inputs)

            # Extract outputs
            if hasattr(model_outputs, "logits"):
                outputs["logits"] = model_outputs.logits
            elif hasattr(model_outputs, "last_hidden_state"):
                 outputs["logits"] = model_outputs.last_hidden_state # Fallback
            
            if hasattr(model_outputs, "loss"):
                 outputs["loss"] = model_outputs.loss

            if hasattr(model_outputs, "hidden_states"):
                outputs["hidden_states"] = model_outputs.hidden_states
            if hasattr(model_outputs, "attentions"):
                outputs["attentions"] = model_outputs.attentions

            # Add past key values for generation
            if hasattr(model_outputs, "past_key_values"):
                outputs["past_key_values"] = model_outputs.past_key_values

            return outputs

        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            raise

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with proper handling for transformer models.
        If model computed loss internally, return it.

        Args:
            outputs: Model outputs dictionary.
            targets: Target tokens.

        Returns:
            Computed loss tensor.
        """
        try:
            # If model already computed loss (e.g. CausalLM with labels), use it
            if "loss" in outputs and outputs["loss"] is not None:
                return outputs["loss"]

            logits = outputs["logits"]

            # Shift logits and targets for language modeling ( Causal LM )
            # logits: [batch, seq_len, vocab]
            # targets: [batch, seq_len]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Compute cross-entropy loss
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

            return loss

        except Exception as e:
            self.logger.error(f"Error computing loss: {e}")
            raise

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate text using the transformer model.
        Delegates to self.model.generate() for efficiency.

        Args:
            input_ids: Input token IDs.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            do_sample: Whether to use sampling.
            **kwargs: Additional generation arguments.

        Returns:
            Generated token IDs.
        """
        self.eval()

        try:
            with torch.no_grad():
                # Use Hugging Face generate method if available
                if hasattr(self.model, "generate"):
                    # attention_mask creation if not provided
                    attention_mask = kwargs.get("attention_mask", None)
                    if attention_mask is None:
                        # Create mask (1 for non-padding)
                        if self.tokenizer and self.tokenizer.pad_token_id is not None:
                             attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
                        else:
                             attention_mask = torch.ones_like(input_ids)
                    
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length + input_ids.shape[1], # HF generate max_length includes input
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                        eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                        **kwargs
                    )
                    return generated
                else:
                    # Fallback to manual loop if model doesn't support generate (unlikely for AutoModelForCausalLM)
                    self.logger.warning("Model does not support .generate(), falling back to manual loop.")
                    return self._generate_manual(input_ids, max_length, temperature, top_p, do_sample, **kwargs)

        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise

    def _generate_manual(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        **kwargs: Any
    ) -> torch.Tensor:
        """Manual generation loop as fallback."""
        generated = input_ids.clone()
        
        for _ in range(max_length):
            outputs = self.forward(generated, **kwargs)
            # Get logits of the last token
            if "logits" in outputs:
                 next_token_logits = outputs["logits"][:, -1, :]
            else:
                 raise ValueError("Model output does not contain logits")

            next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.0)

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if self.tokenizer and next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimized optimizer for transformer training."""
        return AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Get learning rate scheduler with warmup."""
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def apply_gradient_accumulation(
        self, loss: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Apply gradient accumulation.
        Note: This updates optimizer only every gradient_accumulation_steps.
        """
        loss = loss / self.gradient_accumulation_steps

        if self.config.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.config.use_gradient_clipping:
                if self.scaler is not None:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.config.max_grad_norm
                )

            if self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

    def optimize_attention_patterns(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Optimize attention patterns (scaling, dropout).
        
        Args:
            attention_weights: Raw attention weights.

        Returns:
            Optimized attention weights.
        """
        if self.training:
            attention_weights = F.dropout(attention_weights, p=0.1, training=True)

        # Scale attention
        attention_weights = attention_weights / math.sqrt(attention_weights.size(-1))

        return attention_weights

    def get_attention_visualization(
        self, input_ids: torch.Tensor, layer_idx: int = -1
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for visualization.

        Args:
           input_ids: Input tokens.
           layer_idx: Layer index to visualize.

        Returns:
            Averaged attention weights or None.
        """
        self.eval()

        with torch.no_grad():
            # We must enable output_attentions=True in config or forward arg
            # However, standard models often require it in config.
            # Here we just check output.
            outputs = self.forward(input_ids, output_attentions=True)

            if "attentions" in outputs and outputs["attentions"] is not None:
                attention_weights = outputs["attentions"][layer_idx]
                return attention_weights.mean(dim=1)  # Average over heads

        return None

    def apply_quantization(self, quantization_type: str = "int8") -> None:
        """
        Apply model quantization for inference optimization.
        Currently supports 'int8' (dynamic) and 'fp16'.
        """
        try:
            if quantization_type == "int8":
                # Apply dynamic quantization using torch.quantization
                # Using torch.ao.quantization usually preferred in newer versions
                # but torch.quantization is alias.
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear}, dtype=torch.qint8
                )
            elif quantization_type == "fp16":
                self.model = self.model.half()
            
            self.logger.info(f"Applied {quantization_type} quantization")
        except Exception as e:
            self.logger.error(f"Failed to apply quantization {quantization_type}: {e}")

    def get_model_efficiency_metrics(self) -> Dict[str, float]:
        """Get model efficiency metrics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Memory usage estimation (rough generic)
        model_size_mb = total_params * 4 / (1024 * 1024)

        return {
            "total_parameters": float(total_params),
            "trainable_parameters": float(trainable_params),
            "model_size_mb": model_size_mb,
            "efficiency_ratio": (trainable_params / total_params) if total_params > 0 else 0.0,
            "memory_efficiency": (1.0 - (trainable_params / total_params)) if total_params > 0 else 0.0,
        }

