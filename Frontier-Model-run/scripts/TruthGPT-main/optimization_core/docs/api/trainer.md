# Trainer API Reference

The `GenericTrainer` class is the core engine of the TruthGPT system. It handles the full lifecycle of training, validation, and checkpointing.

## `GenericTrainer`

**Location**: `optimization_core/trainers/trainer.py`

```python
class GenericTrainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        train_texts: List[str] | Iterable[str],
        val_texts: List[str] | Iterable[str],
        text_field_max_len: int = 512,
        callbacks: Optional[List[Callback]] = None,
        data_options: Optional[Dict[str, Any]] = None,
    )
```

### Initialization Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `cfg` | `TrainerConfig` | The configuration object containing all hyperparameters and settings. |
| `train_texts` | `Iterable` | List or iterable of strings for training data. |
| `val_texts` | `Iterable` | List or iterable of strings for validation data. |
| `text_field_max_len` | `int` | Maximum sequence length for tokenization (in tokens). Default: `512`. |
| `callbacks` | `List[Callback]` | Optional list of callback objects (e.g., `WandbCallback`, `PrintCallback`). |
| `data_options` | `Dict` | Advanced data loading options (e.g., bucketing strategy). |

### Methods

#### `train()`

Starts the main training loop.

-   **Returns**: `None`
-   **Raises**: `KeyboardInterrupt` if stopped by user, `Exception` on critical failures.

**Description**:
Runs for `cfg.epochs`. Includes logic for:
-   Gradient accumulation
-   Mixed precision scaling (AMP)
-   Gradient clipping
-   NaN/Inf detection
-   Logging and callbacks execution
-   Periodic evaluation and checkpointing

#### `evaluate()`

Runs a full evaluation pass on the validation dataset.

-   **Returns**: `float` (Average Validation Loss)

**Description**:
Switches model to eval mode, disables gradients, and computes loss across the validation set. If `ema_enabled` is true, temporarily swaps in the averaged weights for evaluation.

### Internal Logic

#### Automatic Device Selection
The trainer automatically selects the best available hardware:
1.  **CUDA**: Uses GPU if available (`cfg.device="cuda"` or `"auto"`).
2.  **MPS**: Uses Apple Silicon GPU if available (`cfg.device="mps"` or `"auto"`).
3.  **CPU**: Fallback if no accelerators found.

#### Optimization Integration
-   **LoRA**: If `cfg.lora_enabled=True`, loops through modules and applies Low-Rank Adaptation using `peft`.
-   **Torch Compile**: If `cfg.torch_compile=True`, wraps the model with `torch.compile()`.
