"""
Improved Gradio demo for LLM text generation with better error handling,
input validation, and best practices.
"""
import os
import logging
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast
from typing import Optional, Tuple

from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

# Global model state
MODEL: Optional[AutoModelForCausalLM] = None
TOKENIZER: Optional[AutoTokenizer] = None
DEVICE: Optional[torch.device] = None


def load_model(model_dir: str, device: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load model and tokenizer with proper error handling and device management.
    
    Args:
        model_dir: Path to model directory or HuggingFace model ID
        device: Optional device specification ("cuda", "cpu", "auto")
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    try:
        # Determine device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device_obj = torch.device("cuda")
                dtype = torch.float16  # Use FP16 for efficiency on GPU
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_obj = torch.device("mps")
                dtype = None
            else:
                device_obj = torch.device("cpu")
                dtype = None
        else:
            device_obj = torch.device(device)
            dtype = torch.float16 if device_obj.type == "cuda" else None

        logger.info(f"Loading model from {model_dir} on device {device_obj}")
        
        # Load model and tokenizer
        if os.path.isdir(model_dir) and os.path.exists(model_dir):
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map="auto" if device_obj.type == "cuda" else None,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        else:
            # Try loading from HuggingFace Hub
            logger.info(f"Loading model from HuggingFace Hub: {model_dir}")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map="auto" if device_obj.type == "cuda" else None,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Move to device if not using device_map
        if not hasattr(model, "hf_device_map"):
            model.to(device_obj)
        
        # Enable cache for generation
        if hasattr(model, "config"):
            try:
                model.config.use_cache = True
            except Exception:
                pass
        
        model.eval()
        logger.info("Model loaded successfully")
        return model, tokenizer, device_obj
        
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise


def validate_inputs(prompt: str, max_new_tokens: int, temperature: float) -> Tuple[bool, Optional[str]]:
    """
    Validate input parameters for generation.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        return False, "Prompt must be a non-empty string"
    
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1 or max_new_tokens > 2048:
        return False, "max_new_tokens must be an integer between 1 and 2048"
    
    if not isinstance(temperature, (int, float)) or temperature <= 0 or temperature > 2.0:
        return False, "Temperature must be a positive number between 0 and 2.0"
    
    return True, None


def generate_response(
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True
) -> str:
    """
    Generate text response from prompt with improved error handling.
    
    Args:
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty factor
        do_sample: Whether to use sampling
    
    Returns:
        Generated text or error message
    """
    global MODEL, TOKENIZER, DEVICE
    
    # Check if model is loaded
    if MODEL is None or TOKENIZER is None or DEVICE is None:
        error_msg = "Model not loaded. Please check the model checkpoint path."
        logger.error(error_msg)
        return error_msg
    
    # Validate inputs
    is_valid, error_msg = validate_inputs(prompt, max_new_tokens, temperature)
    if not is_valid:
        logger.warning(f"Invalid input: {error_msg}")
        return f"Error: {error_msg}"
    
    try:
        # Tokenize input
        inputs = TOKENIZER(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Limit input length
        ).to(DEVICE)
        
        # Generate with autocast for mixed precision if on GPU
        with torch.no_grad():
            if DEVICE.type == "cuda":
                with autocast():
                    output_ids = MODEL.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p if do_sample else None,
                        top_k=top_k if do_sample else None,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=TOKENIZER.eos_token_id,
                        eos_token_id=TOKENIZER.eos_token_id,
                        use_cache=True,
                    )
            else:
                output_ids = MODEL.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p if do_sample else None,
                    top_k=top_k if do_sample else None,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=TOKENIZER.eos_token_id,
                    eos_token_id=TOKENIZER.eos_token_id,
                    use_cache=True,
                )
        
        # Decode output
        generated_text = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
        
        # Return only the newly generated part (remove input prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        logger.debug(f"Generated {len(generated_text.split())} words from prompt")
        return generated_text if generated_text else prompt
        
    except torch.cuda.OutOfMemoryError:
        error_msg = "GPU out of memory. Try reducing max_new_tokens or using a smaller model."
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"


def create_interface() -> gr.Interface:
    """Create Gradio interface with improved UI and error handling."""
    
    # Load model on startup
    model_path = os.environ.get("LLM_CHECKPOINT", "gpt2")
    try:
        global MODEL, TOKENIZER, DEVICE
        MODEL, TOKENIZER, DEVICE = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        # Continue with None model - will show error in UI
    
    interface = gr.Interface(
        fn=generate_response,
        inputs=[
            gr.Textbox(
                label="Prompt",
                lines=4,
                value="The theory of transformers is",
                placeholder="Enter your prompt here...",
                info="Input text prompt for generation"
            ),
            gr.Slider(
                minimum=1,
                maximum=512,
                value=64,
                step=1,
                label="Max New Tokens",
                info="Maximum number of tokens to generate (1-512)"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.8,
                step=0.05,
                label="Temperature",
                info="Sampling temperature (0.1-2.0)"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
                step=0.05,
                label="Top-p (Nucleus Sampling)",
                info="Nucleus sampling parameter"
            ),
            gr.Slider(
                minimum=1,
                maximum=100,
                value=50,
                step=1,
                label="Top-k",
                info="Top-k sampling parameter"
            ),
            gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.1,
                step=0.05,
                label="Repetition Penalty",
                info="Penalty for repetition (1.0 = no penalty)"
            ),
            gr.Checkbox(
                value=True,
                label="Use Sampling",
                info="Enable sampling (disable for greedy decoding)"
            ),
        ],
        outputs=[
            gr.Textbox(
                label="Generated Text",
                lines=8,
                show_copy_button=True
            )
        ],
        title="LLM Text Generation Demo",
        description=(
            "Generate text using a fine-tuned language model. "
            "The model is loaded from checkpoint or HuggingFace Hub."
        ),
        examples=[
            ["The theory of transformers is"],
            ["In the future, artificial intelligence will"],
            ["Once upon a time, in a distant galaxy"],
        ],
        cache_examples=False,
        theme=gr.themes.Soft(),
    )
    
    return interface


if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()
    demo.queue(
        concurrency_count=2,
        max_size=10,
        default_concurrency_limit=2
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

