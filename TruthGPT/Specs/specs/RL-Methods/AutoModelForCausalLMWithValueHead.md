## 🧩 AutoModelForCausalLMWithValueHead Module

An autoregressive model with a value head in addition to the language model head. This class inherits from `~trl.PreTrainedModelWrapper` and wraps a `transformers.PreTrainedModel` class. The wrapper class supports classic functions such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped model, simply manipulate the `pretrained_model` attribute of this class.

### ✨ Key Features

*   **Wrapper Design**: Combines a pre-trained autoregressive language model (e.g., from `transformers.AutoModelForCausalLM`) with an additional value head, commonly used in reinforcement learning setups like PPO.
*   **TRL Integration**: Built upon `trl.PreTrainedModelWrapper`, ensuring seamless integration with the TRL (Transformer Reinforcement Learning) library and its components.
*   **Configurable Value Head**: The characteristics of the value head can be customized during initialization using specific arguments (see Parameters section).
*   **Standard Hugging Face API**: Supports essential `transformers` functionalities such as `from_pretrained` for loading models, `push_to_hub` for sharing, and `generate` for text generation.

### 🐍 Python Class Overview

```python
# Based on trl.models.AutoModelForCausalLMWithValueHead
# PreTrainedModelWrapper and AutoModelForCausalLM are typically from trl and transformers

class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    r"""
    An autoregressive model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class.
    """
    # Class attributes:
    transformers_parent_class: type # e.g., transformers.AutoModelForCausalLM
    supported_args: tuple = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
    )

    def __init__(self, pretrained_model, **kwargs):
        """
        Initializes the model by wrapping the `pretrained_model`.
        The `ValueHead` is also initialized here, configured by `supported_args` passed via `kwargs`.
        """
        super().__init__(pretrained_model, **kwargs)
        # Actual ValueHead initialization happens within the PreTrainedModelWrapper or this class

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # ... other arguments accepted by the base model (e.g., past_key_values, inputs_embeds)
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], object]: # Return type is often a dataclass like CausalLMOutputWithValue
        """
        Performs a forward pass through the base language model and the value head.

        Returns:
            A tuple or a structured object containing:
            - Language model outputs (e.g., logits).
            - Value estimates from the value head.
            - Optionally, loss if labels are provided.
        """
        # Internally, this method calls the base model, then passes its hidden states
        # (or other suitable outputs) to the value head.
        pass
```

### 📥 Parameters (for ValueHead Configuration)

These parameters are typically passed as `**kwargs` to the constructor or `from_pretrained` method and are used to configure the `ValueHead` integrated into the model.

| Parameter                  | Type    | Description                                                                                                                               |
|----------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `summary_dropout_prob`     | `float` | Dropout probability for the linear layer in the `ValueHead`. Optional, defaults to `None` (no dropout).                                   |
| `v_head_initializer_range` | `float` | The initializer range for the weights of the `ValueHead` if a specific initialization strategy (e.g., "normal") is selected. Optional, defaults to `0.2`. |
| `v_head_init_strategy`     | `str`   | The initialization strategy for the `ValueHead`. Supported strategies: `None` (default PyTorch initialization) or `"normal"`. Optional, defaults to `None`. |

### ↔️ Input / Output Shape

**Input:**

*   `input_ids (torch.LongTensor)`: Shape `(batch_size, seq_len)`
    *   Token indices representing the input sequence.
*   `attention_mask (Optional[torch.Tensor])`: Shape `(batch_size, seq_len)`
    *   Mask indicating which tokens should be attended to (1 for attended, 0 for padding).
*   `labels (Optional[torch.LongTensor])`: Shape `(batch_size, seq_len)`
    *   Labels for causal language modeling loss calculation.
*   `past_key_values (Optional[Tuple[Tuple[torch.Tensor]]])`:
    *   Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding.
*   `inputs_embeds (Optional[torch.FloatTensor])`: Shape `(batch_size, seq_len, hidden_size)`
    *   Optionally, use embedded inputs directly instead of `input_ids`.
*   Other arguments: Any other arguments accepted by the underlying Hugging Face `transformers.PreTrainedModel`.

**Output:**

The output format can vary (e.g., a tuple or a `transformers.modeling_outputs.CausalLMOutputWithPast` like object, often extended with a value field, like `CausalLMOutputWithValue`). Generally, it includes:

*   `logits (torch.FloatTensor)`: Shape `(batch_size, seq_len, vocab_size)`
    *   Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
*   `value (torch.FloatTensor)`: Shape `(batch_size, seq_len, 1)` or `(batch_size, 1)`
    *   Value estimates from the value head. The exact shape depends on the `ValueHead` implementation (e.g., per token or pooled).
*   `loss (Optional[torch.FloatTensor])`: Scalar
    *   Causal language modeling loss, returned when `labels` are provided.
*   `past_key_values (Optional[Tuple[Tuple[torch.Tensor]]])`:
    *   Returned if `use_cache=True` is passed; contains pre-computed key/value states for faster generation.
*   `hidden_states (Optional[Tuple[torch.FloatTensor]])`:
    *   Hidden states of the model at the output of each layer plus the initial embedding outputs. Returned if `output_hidden_states=True`.
*   `attentions (Optional[Tuple[torch.FloatTensor]])`:
    *   Attentions weights after the attention softmax, used to inspect the model's attention patterns. Returned if `output_attentions=True`. 